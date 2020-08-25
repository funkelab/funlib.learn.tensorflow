import logging

import numpy as np
import tensorflow as tf

from .layers import conv_pass, downsample, upsample
from .utils import (crop_spatial_temporal, crop_to_factor,
                    get_number_of_tf_variables)

logger = logging.getLogger(__name__)


def unet(
        fmaps_in,
        num_fmaps,
        fmap_inc_factors,
        downsample_factors,
        fmap_dec_factors=None,
        kernel_size_down=None,
        kernel_size_up=None,
        activation='relu',
        padding='valid',
        upsampling='transposed_conv',
        constant_upsample=None,
        is_training=None,
        use_batchnorm=False,
        shortcut=False,
        layer=0,
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1),
        num_fmaps_out=None,
        num_heads=1,
        return_summaries=False):
    '''Create a U-Net::

        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...

    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.

    The U-Net expects 3D or 4D tensors shaped like::

        ``(batch=1, channels, [length,] depth, height, width)``.

    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution. It will perfrom 4D convolutions as
    long as ``length`` is greater than 1. As soon as ``length`` is 1 due to a
    valid convolution, the time dimension will be dropped and tensors with
    ``(b, c, z, y, x)`` will be use (and returned) from there on.

    Args:

        fmaps_in:

            The input tensor.

        num_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps. Stored in the ``channels``
            dimension.

        fmap_inc_factors (``int`` or list of ``int``):

            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.

        fmap_dec_factors (``int`` or list of ``int``):

            By how much to divide the number of feature maps between layers
            in the decoder path.
            If not set, reverse of fmap_inc_factors will be used.

        downsample_factors:

            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.

        kernel_size_down (optional):

            List of lists of kernel sizes. The number of sizes in a list
            determines the number of convolutional layers in the corresponding
            level of the build on the left side. Kernel sizes can be given as
            tuples or integer. If not given, each convolutional pass will
            consist of two (3x)3x3 convolutions.

        kernel_size_up (optional):

            List of lists of kernel sizes. The number of sizes in a list
            determines the number of convolutional layers in the corresponding
            level of the build on the right side. Within one of the lists going
            from left to right. Kernel sizes can be given as tuples or integer.
            If not given, each convolutional pass will consist of two (3x)3x3
            convolutions.

        activation:

            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

        padding (``string``):

            Which kind of padding to use, 'valid' or 'same' (case-insensitive)
        upsampling (``string``):

            Type of upsampling used, one of
            ['transposed_conv', 'resize_conv', 'uniform_transposed_conv']
            transposed_conv: use a normal transposed convolution
            resize_conv: replicate each pixel factors times followed
            by a 1x1 conv
            uniform_transposed_conv: use a transposed convolution with
            a factors sized filter with a single, learnable value at all
            positions
        constant_upsample (``bool``, optional):

            Whether to restrict the transpose convolution kernels to be
            constant values. This might help to reduce checker board artifacts.
            (deprecated, use upsampling)

        is_training:

            Boolean or tf.placeholder to set batchnorm and dropout to
            training or test mode

        use_batchnorm:

            Whether to use batch norm layers after convolution

        shortcut:

            Whether to add residual shortcut/skip connections

        layer:

            Used internally to build the U-Net recursively.
        fov:

            Initial field of view in physical units

        voxel_size:

            Size of a voxel in the input data, in physical units

        num_fmaps_out:

            If given, specifies the number of output fmaps of the U-Net.
            Setting this number ensures that the upper most layer, right side
            has at least this number of fmaps.

        num_heads:

            Number of decoders. The resulting U-Net has one single encoder
            path and num_heads decoder paths. This is useful in a multi-task
            learning context.

        return_summaries:

            Whether to return tf/tensorboard summaries

    '''
    num_var_start = get_number_of_tf_variables()
    prefix = "    "*layer
    logger.info(prefix + "Creating U-Net layer %i", layer)
    logger.info(prefix + "f_in: %s", fmaps_in.shape)
    if isinstance(fmap_inc_factors, int):
        fmap_inc_factors = [fmap_inc_factors]*len(downsample_factors)

    # by default, create 2 3x3x3 convolutions per layer
    if kernel_size_down is None:
        kernel_size_down = [[3, 3]]*(len(downsample_factors) + 1)
    if kernel_size_up is None:
        kernel_size_up = [[3, 3]]*len(downsample_factors)

    assert (
        len(fmap_inc_factors) ==
        len(downsample_factors) ==
        len(kernel_size_down) - 1 ==
        len(kernel_size_up))

    # convolve
    f_left, fov = conv_pass(
        fmaps_in,
        kernel_sizes=kernel_size_down[layer],
        num_fmaps=num_fmaps,
        activation=activation,
        padding=padding,
        is_training=is_training,
        use_batchnorm=use_batchnorm,
        shortcut=shortcut,
        name='unet_layer_%i_left' % layer,
        fov=fov,
        voxel_size=voxel_size)

    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))

    num_var_end = get_number_of_tf_variables()
    var_added = num_var_end - num_var_start
    if bottom_layer:
        logger.info(prefix + "bottom layer")
        logger.info(prefix + "f_out: %s", f_left.shape)
        if num_heads > 1:
            f_left = [f_left] * num_heads
        logger.info(prefix + 'number of variables added: %i, '
                    'new total: %i', var_added, num_var_end)
        return f_left, fov, voxel_size

    # downsample
    g_in, voxel_size = downsample(
        f_left,
        downsample_factors[layer],
        padding=padding,
        name='unet_down_%i_to_%i' % (layer, layer + 1),
        voxel_size=voxel_size)

    logger.info(prefix + 'number of variables added: %i, '
                'new total: %i', var_added, num_var_end)
    # recursive U-net
    g_outs, fov, voxel_size = unet(
        g_in,
        num_fmaps=int(num_fmaps*fmap_inc_factors[layer]),
        fmap_inc_factors=fmap_inc_factors,
        downsample_factors=downsample_factors,
        fmap_dec_factors=fmap_dec_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        activation=activation,
        padding=padding,
        upsampling=upsampling,
        constant_upsample=constant_upsample,
        is_training=is_training,
        use_batchnorm=use_batchnorm,
        shortcut=shortcut,
        layer=layer+1,
        fov=fov,
        voxel_size=voxel_size,
        num_heads=num_heads)
    if num_heads == 1:
        g_outs = [g_outs]

    if fmap_dec_factors is not None:
        num_fmaps_up = int(num_fmaps * np.prod(fmap_inc_factors[layer:])
                           / np.prod(fmap_dec_factors[layer:]))
    else:
        num_fmaps_up = num_fmaps

    # For Multi-Headed UNet: Create this path multiple times.
    f_outs = []
    for head_num, g_out in enumerate(g_outs):
        num_var_start = get_number_of_tf_variables()
        with tf.variable_scope('decoder_%i_layer_%i' % (head_num, layer)):
            if num_heads > 1:
                logger.info(prefix + 'head number: %i', head_num)
            logger.info(prefix + "g_out: %s", g_out.shape)
            # upsample
            g_out_upsampled, voxel_size = upsample(
                g_out,
                downsample_factors[layer],
                num_fmaps_up,
                upsampling=upsampling,
                activation=activation,
                padding=padding,
                name='unet_up_%i_to_%i' % (layer + 1, layer),
                voxel_size=voxel_size,
                constant_upsample=constant_upsample)

            logger.info(prefix + "g_out_upsampled: %s",
                        g_out_upsampled.shape)

            # ensure translation equivariance with stride of product of
            # previous downsample factors
            factor_product = None
            for factor in downsample_factors[layer:]:
                if factor_product is None:
                    factor_product = list(factor)
                else:
                    factor_product = list(
                        f*ff
                        for f, ff in zip(factor, factor_product))
            g_out_upsampled = crop_to_factor(
                g_out_upsampled,
                factor=factor_product,
                kernel_sizes=kernel_size_up[layer])

            logger.info(prefix + "g_out_upsampled_cropped: %s",
                        g_out_upsampled.shape)

            # copy-crop
            f_left_cropped = crop_spatial_temporal(
                f_left,
                g_out_upsampled.get_shape().as_list())

            logger.info(prefix + "f_left_cropped: %s", f_left_cropped.shape)

            # concatenate along channel dimension
            f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

            logger.info(prefix + "f_right: %s", f_right.shape)

            if layer == 0 and num_fmaps_out is not None:
                num_fmaps_up = max(num_fmaps_out, num_fmaps_up)

            # convolve
            f_out, fov = conv_pass(
                f_right,
                kernel_sizes=kernel_size_up[layer],
                num_fmaps=num_fmaps_up,
                activation=activation,
                padding=padding,
                is_training=is_training,
                use_batchnorm=use_batchnorm,
                shortcut=shortcut,
                name='unet_layer_%i_right' % (layer),
                fov=fov,
                voxel_size=voxel_size)

            logger.info(prefix + "f_out: %s", f_out.shape)
            f_outs.append(f_out)
            num_var_end = get_number_of_tf_variables()
            var_added = num_var_end - num_var_start
            logger.info(prefix + 'number of variables added: %i, '
                        'new total: %i', var_added, num_var_end)
    if num_heads == 1:
        f_outs = f_outs[0]  # Backwards compatibility.

    if return_summaries:
        summaries = add_summaries()
        return f_outs, fov, voxel_size, summaries

    return f_outs, fov, voxel_size
