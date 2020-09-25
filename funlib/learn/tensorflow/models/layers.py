import logging

import numpy as np
import tensorflow as tf

from .conv4d import conv4d
from .utils import crop

logger = logging.getLogger(__name__)


def conv(fmaps_in,
         num_fmaps,
         kernel_size,
         activation='relu',
         padding='valid',
         use_bias=True,
         strides=1,
         name='conv',
         fov=(1, 1, 1),
         voxel_size=(1, 1, 1)):
    '''Create a single nd convolution::

        If padding is 'valid', each convolution will decrease the size
        of the feature maps by ``kernel_size-1``.

    Args:

        fmaps_in:

            The input tensor of shape ``(batch_size, channels, [length,] depth,
            height, width)``.

        num_fmaps:

            The number of feature maps to produce with each convolution.

        kernel_size:

            Size of the kernel to use. Forwarded to tf.layers.convNd.


        activation:

            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

        padding:

            Which kind of padding to use, 'valid' or 'same' (case-insensitive)

        use_bias:

            Whether the layer uses a bias variable

        strides:

            Specifiy strides of the convolution

        name:

            Base name for the conv layer.

        fov:

            Field of view of fmaps_in, in physical units.

        voxel_size:

            Size of a voxel in the input data, in physical units.

    Returns:

        (fmaps, fov):

            The feature maps after the last convolution, and a tuple
            representing the field of view.
    '''

    in_shape = tuple(fmaps_in.get_shape().as_list())
    # Explicitly handle number of dimensions
    if len(in_shape) == 6:
        conv_op = conv4d
    elif len(in_shape) == 5:
        conv_op = tf.layers.conv3d
    elif len(in_shape) == 4:
        conv_op = tf.layers.conv2d
    else:
        raise RuntimeError(
            "Input tensor of shape %s not supported" % (in_shape,))

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]*(len(in_shape) - 2)

    fov = tuple(
        f + (k - 1)*vs
        for f, k, vs
        in zip(fov, kernel_size, voxel_size)
    )

    fmaps = conv_op(
        inputs=fmaps_in,
        filters=num_fmaps,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=use_bias,
        padding=padding,
        data_format='channels_first',
        activation=activation,
        name=name)
    logger.info("%s", fmaps)

    out_shape = tuple(fmaps.get_shape().as_list())

    # eliminate t dimension if length is 1
    if len(out_shape) == 6:
        length = out_shape[2]
        if length == 1:
            fmaps = tf.squeeze(fmaps, axis=2)

    return fmaps, fov


def conv_pass(
        fmaps_in,
        kernel_sizes,
        num_fmaps,
        activation='relu',
        padding='valid',
        strides=1,
        is_training=None,
        use_batchnorm=False,
        shortcut=False,
        name='conv_pass',
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1)):
    '''Create a convolution pass::
        f_in --> f_1 --> ... --> f_n
    where each ``-->`` is a convolution, optionally followed by a batch
    normalization layer and a(non-linear) activation function.
    One convolution will be performed for each entry in
    ``kernel_sizes``.
    Args:
        fmaps_in:
            The input tensor of shape
            ``(batch_size, channels, [length,] depth, height, width)``.
        kernel_sizes:
            Sizes of the kernels to use. Forwarded to tf.layers.convNd.
        num_fmaps:
            The number of feature maps to produce with each convolution.
        activation:
            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

        padding:
            Which kind of padding to use, 'valid' or 'same' (case-insensitive)
        strides:
            Specifiy strides of the convolution
        is_training:
            Boolean or tf.placeholder to set batchnorm and dropout to
            training or test mode
        use_batchnorm:
            Whether to use batch norm layers after convolution
        shortcut:
            Whether to add residual shortcut/skip connections
        name:
            Base name for the conv layer.
        fov:
            Field of view of fmaps_in, in physical units.
        voxel_size (``tuple`` of ``int``, optional):
            Voxel size of the input feature maps. Used to compute the voxel
            size of the output.
    Returns:
        (fmaps, fov):
            The new feature maps and a tuple representing the field of view
    '''

    in_shape = fmaps_in.get_shape().as_list()
    num_fmaps_in = in_shape[1]
    fmaps = fmaps_in

    if use_batchnorm:
        conv_activation = None
        use_bias = False
    else:
        conv_activation = activation
        use_bias = True

    for i, kernel_size in enumerate(kernel_sizes):
        fmaps, fov = conv(fmaps, num_fmaps, kernel_size,
                          activation=activation,
                          padding=padding,
                          use_bias=use_bias,
                          name=name + "_%i" % i,
                          fov=fov, voxel_size=voxel_size)

        if use_batchnorm:
            fmaps = tf.layers.batch_normalization(
                fmaps,
                axis=1,
                training=is_training,
                epsilon=1.0001e-5,
                name=name + '%i_bn' % i)
            logger.info("%s", fmaps)
            fmaps = tf.keras.activations.get(activation)(fmaps)
                # fmaps, name=name + '%i_relu' % i)
            logger.info("%s", fmaps)

    if shortcut:
        out_shape = fmaps.get_shape().as_list()
        if out_shape != in_shape:
            if len(out_shape) != len(in_shape):
                raise RuntimeWarning(
                    "residual skip connections not implemented yet "
                    "for temporal convolutions")
            fmaps_in = crop(fmaps_in, out_shape)
        fmaps = tf.add(fmaps, fmaps_in, name=name + '_shortcut')
        logger.info("%s", fmaps)


    return fmaps, fov


def basic_res_block(fmaps_in,
                    num_fmaps,
                    num_fmaps_out,
                    activation='relu',
                    padding='same',
                    strides=1,
                    is_training=None,
                    use_batchnorm=False,
                    is_first_block=False,
                    name='basic_res_block',
                    fov=(1, 1, 1),
                    voxel_size=(1, 1, 1)):
    '''Create a basic residual block (preact, resnetv2):
    Args:
        fmaps_in:
            The input tensor of shape
            ``(batch_size, channels, [length,] depth, height, width)``.
        num_fmaps:
            The number of feature maps to produce with each convolution.
        fov:
            Field of view of fmaps_in, in physical units.
        name:
            Base name for the conv layer.
    Returns:
        (fmaps, fov):
            The feature maps after the last convolution, and a tuple
            representing the field of view
    '''

    fmaps = fmaps_in
    if use_batchnorm:
        conv_activation = None
        use_bias = False
    else:
        conv_activation = activation
        use_bias = True

    kernel_sizes = [3, 3]
    for i, kernel_size in enumerate(kernel_sizes):
        if use_batchnorm and not is_first_block:
            fmaps = tf.layers.batch_normalization(
                fmaps,
                axis=1,
                training=is_training,
                epsilon=1.0001e-5,
                name=name + '_%i_bn' % i)
            logger.info("%s", fmaps)
            fmaps = tf.keras.activations.get(activation)(fmaps)
                # fmaps, name=name + '_%i_act' % i)
            logger.info("%s", fmaps)

        fmaps, fov = conv(fmaps,
                          num_fmaps if i == 0 else num_fmaps_out,
                          kernel_size,
                          activation=activation,
                          padding=padding,
                          use_bias=use_bias,
                          # strides=strides if i == 0 else 1,
                          strides=strides if i == 1 else 1,
                          name=name + "_%i" % i,
                          fov=fov, voxel_size=voxel_size,)

    return fmaps, fov


def bottleneck_res_block(fmaps_in,
                         num_fmaps,
                         num_fmaps_out,
                         activation='relu',
                         padding='same',
                         strides=1,
                         is_training=None,
                         use_batchnorm=False,
                         is_first_block=False,
                         name='bottleneck_res_block',
                         fov=(1, 1, 1),
                         voxel_size=(1, 1, 1)):
    '''Create a bottleneck residual block (preact, resnetv2):
    Args:
        fmaps_in:
            The input tensor of shape
            ``(batch_size, channels, [length,] depth, height, width)``.
        num_fmaps:
            The number of feature maps to use in inner convolution.
        num_fmaps_out:
            The number of feature maps to output.
        activation:
            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

        padding:
            Which kind of padding to use, 'valid' or 'same' (case-insensitive)
        strides:
            Specifiy strides of the convolution
        is_training:
            Boolean or tf.placeholder to set batchnorm and dropout to
            training or test mode
        use_batchnorm:
            Whether to use batch norm layers after convolution
        is_first_block:
            Whether this is the first block in the network
        name:
            Base name for the conv layer.
        fov:
            Field of view of fmaps_in, in physical units.
        voxel_size:
            Voxel size of the input feature maps. Used to compute the voxel
            size of the output.
    Returns:
        (fmaps, fov):
            The feature maps after the last convolution, and a tuple
            representing the field of view
    '''

    fmaps = fmaps_in
    if use_batchnorm:
        conv_activation = None
        use_bias = False
    else:
        conv_activation = activation
        use_bias = True

    # bottleneck in
    if use_batchnorm and not is_first_block:
        fmaps = tf.layers.batch_normalization(
            fmaps,
            axis=1,
            training=is_training,
            epsilon=1.0001e-5,
            name=name + '_in_bn')
        logger.info("%s", fmaps)
        fmaps = tf.keras.activations.get(activation)(fmaps)
            # fmaps, name=name + '_in_act')
        logger.info("%s", fmaps)

    fmaps, fov = conv(
        fmaps, num_fmaps, 1,
        activation=activation,
        padding=padding,
        use_bias=use_bias,
        strides=1,
        name=name + '_in_conv',
        fov=fov, voxel_size=voxel_size)

    # bottleneck
    if use_batchnorm:
        fmaps = tf.layers.batch_normalization(
            fmaps,
            axis=1,
            training=is_training,
            epsilon=1.0001e-5,
            name=name + '_bn')
        logger.info("%s", fmaps)
        fmaps = tf.keras.activations.get(activation)(fmaps)
            # fmaps, name=name + '_act')
        logger.info("%s", fmaps)

    fmaps, fov = conv(fmaps, num_fmaps, 3,
                      activation=activation,
                      padding=padding,
                      use_bias=use_bias,
                      strides=strides,
                      name=name + "_conv",
                      fov=fov, voxel_size=voxel_size,)

    # bottleneck out
    if use_batchnorm:
        fmaps = tf.layers.batch_normalization(
            fmaps,
            axis=1,
            training=is_training,
            epsilon=1.0001e-5,
            name=name + '_out_bn')
        logger.info("%s", fmaps)
        fmaps = tf.keras.activations.get(activation)(fmaps)
            # fmaps, name=name + '_out_act')
        logger.info("%s", fmaps)

    fmaps, fov = conv(fmaps, num_fmaps_out, 1,
                      activation=activation,
                      padding=padding,
                      use_bias=use_bias,
                      strides=1,
                      name=name + '_out_conv',
                      fov=fov, voxel_size=voxel_size)

    return fmaps, fov


def downsample(
        fmaps_in,
        factors,
        strides=None,
        padding='valid',
        name='down',
        voxel_size=(1, 1, 1)):
    in_shape = fmaps_in.get_shape().as_list()

    # Explicitly handle number of dimensions
    is_4d = len(in_shape) == 6
    if len(in_shape) == 4:
        # 2d
        pool_op = tf.layers.max_pooling2d
        num_dims = 2
    else:
        # 3d, 4d
        pool_op = tf.layers.max_pooling3d
        num_dims = 3

    if is_4d:
        orig_in_shape = in_shape
        # store time dimension in channels
        fmaps_in = tf.reshape(fmaps_in, (
            in_shape[0] if in_shape[0] is not None else -1,
            in_shape[1]*in_shape[2],
            in_shape[3],
            in_shape[4],
            in_shape[5]))
        in_shape = fmaps_in.get_shape().as_list()

    if isinstance(factors, int):
        factors = (len(in_shape)-2)*[factors]

    if strides is None:
        strides = factors
    else:
        if isinstance(strides, int):
            strides = (len(in_shape)-2)*[strides]

    voxel_size = tuple(vs*st for vs, st in
                       zip(voxel_size, strides[-len(voxel_size):]))

    if not np.all(np.array(in_shape[2:]) % np.array(factors) == 0):
        raise RuntimeWarning(
            "Input shape %s is not evenly divisible by downsample factor %s." %
            (in_shape[2:], factors))

    fmaps = pool_op(
        fmaps_in,
        pool_size=factors[-num_dims:],
        strides=strides[-num_dims:],
        padding=padding,
        data_format='channels_first',
        name=name,
    )

    if is_4d:
        out_shape = fmaps.get_shape().as_list()

        # restore time dimension
        fmaps = tf.reshape(fmaps, (
            orig_in_shape[0] if orig_in_shape[0] is not None else -1,
            orig_in_shape[1],
            orig_in_shape[2],
            out_shape[2],
            out_shape[3],
            out_shape[4]))

    return fmaps, voxel_size


def upsample(
        fmaps_in,
        factors,
        num_fmaps,
        upsampling=None,
        activation='relu',
        padding='valid',
        name='up',
        voxel_size=(1, 1, 1),
        constant_upsample=None):
    '''Upsample feature maps with the given factors using a transposed
    convolution.

    Args:

        fmaps_in (tensor):

            The input feature maps of shape `(b, c, d, h, w)`. `c` is the
            number of channels (number of feature maps).

        factors (``tuple`` of ``int``):

            The spatial upsampling factors as `(f_z, f_y, f_x)`.

        num_fmaps (``int``):

            The number of output feature maps.

        upsampling (``string``):

            Type of upsampling used, one of
            ['transposed_conv', 'resize_conv', 'uniform_transposed_conv']
            transposed_conv: use a normal transposed convolution
            resize_conv: replicate each pixel factors times followed
            by a 1x1 conv
            uniform_transposed_conv: use a transposed convolution with
            a factors sized filter with a single, learnable value at all
            positions

        activation (``string``):

            Which activation function to use.

        padding (``string``):

            Which kind of padding to use, 'valid' or 'same' (case-insensitive)

        name (``string``):

            Name of the operator.

        voxel_size (``tuple`` of ``int``, optional):

            Voxel size of the input feature maps. Used to compute the voxel
            size of the output.

        constant_upsample (``bool``, optional):

            Whether to restrict the transpose convolution kernels to be
            constant values. This might help to reduce checker board artifacts.
            (deprecated, use upsampling)

    Returns:

        `(fmaps, voxel_size)`, with `fmaps` of shape `(b, num_fmaps, d*f_z,
        h*f_y, w*f_x)`.
    '''

    if constant_upsample is not None:
        logger.info(
            "upsample layer: constant_upsample is deprecated, "
            "please use the upsampling argument: "
            "constant_upsample = True: upsampling = 'uniform_transposed_conv"
            "constant_upsample = False: upsampling = 'transposed_conv")
        if upsampling is None:
            if constant_upsample:
                upsampling = "uniform_transposed_conv"
            else:
                upsampling = "transposed_conv"

    # Explicitly handle number of dimensions
    if len(fmaps_in.get_shape().as_list()) == 4:
        # 2d
        spatial_dims = (1, 1)
        transp_op = tf.nn.conv2d_transpose
        transp_layer = tf.layers.conv2d_transpose
        resize_op = tf.keras.backend.resize_images
    else:
        # 3d
        spatial_dims = (1, 1, 1)
        transp_op = tf.nn.conv3d_transpose
        transp_layer = tf.layers.conv3d_transpose
        resize_op = tf.keras.backend.resize_volumes

    in_shape = tuple(fmaps_in.get_shape().as_list())
    if isinstance(factors, int):
        factors = (len(in_shape)-2)*[factors]

    voxel_size = tuple(vs/fac for vs, fac
                       in zip(voxel_size, factors[-len(voxel_size):]))

    if upsampling == "uniform_transposed_conv":

        num_fmaps_in = in_shape[1]
        num_fmaps_out = num_fmaps
        out_shape = (
            in_shape[0],
            num_fmaps_out) + tuple(s*f for s, f in zip(in_shape[2:], factors))

        # (num_fmaps_out * num_fmaps_in)
        kernel_variables = tf.get_variable(
            name + '_kernel_variables',
            (num_fmaps_out * num_fmaps_in,),
            dtype=tf.float32,
            trainable=True)

        # ((1), 1, 1, num_fmaps_out, num_fmaps_in)
        kernel_variables = tf.reshape(
            kernel_variables,
            spatial_dims + (num_fmaps_out, num_fmaps_in))

        # ((f_z, )f_y, f_x, num_fmaps_out, num_fmaps_in)
        expanded = tf.expand_dims(kernel_variables, -1)
        tiled = tf.tile(expanded,
                        multiples=(1,) + tuple(factors) + (1, 1))
        constant_upsample_filter = tf.reshape(
            tiled,
            tf.shape(kernel_variables) * (tuple(factors) + (1, 1)))

        fmaps = transp_op(
            fmaps_in,
            filter=constant_upsample_filter,
            output_shape=out_shape,
            strides=(1, 1) + tuple(factors),
            padding=padding,
            data_format='channels_first',
            name=name + "_uniform_transp")

    elif upsampling == "transposed_conv":
        fmaps = transp_layer(
            fmaps_in,
            filters=num_fmaps,
            kernel_size=factors,
            strides=factors,
            padding=padding,
            data_format='channels_first',
            name=name + "_transp_conv",
        )
    elif upsampling == "resize_conv":
        fmaps = resize_op(
            fmaps_in,
            *factors,
            "channels_first")
        fmaps, _ = conv(fmaps, num_fmaps, 1,
                        activation=None,
                        padding=padding,
                        strides=1,
                        name=name + "_resize_conv")

    logger.info("%s", fmaps)

    fmaps = tf.keras.activations.get(activation)(fmaps)
    logger.info("%s", fmaps)

    return fmaps, voxel_size
