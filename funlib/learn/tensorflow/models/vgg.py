import logging

import numpy as np
import tensorflow as tf

from .layers import conv, conv_pass, downsample
from .utils import (add_summaries,
                    get_number_of_tf_variables,
                    global_average_pool)

logger = logging.getLogger(__name__)


def vgg(fmaps_in,
        *,  # this indicates that all following arguments are keyword arguments
        kernel_sizes,
        num_fmaps,
        fmap_inc_factors,
        downsample_factors,
        fc_size,
        num_classes,
        activation='relu',
        padding='same',
        make_iso=False,
        merge_time_voxel_size=None,
        is_training=None,
        use_batchnorm=False,
        use_conv4d=False,
        global_pool=True,
        dropout=False,
        voxel_size=(1, 1, 1)):

    ''' Create a VGG-Net:

    Args:
        fmaps_in:

            The input tensor of shape ``(batch_size, channels, [length,] depth,
            height, width)``.

        kernel_sizes:

            List of lists of kernel sizes. The number of sizes in a list
            determines the number of convolutional layers in the corresponding
            level of the build. Kernel sizes can be given as tuples or integer.

        num_fmaps:

            The number of feature maps to produce with each convolution.

        fmap_inc_factors:

            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.

        downsample_factors:

            List of lists ``[z, y, x]`` to use to down-sample the
            feature maps between layers.

        fc_size:

            Size of fully connected layer
            (implementation-wise a 1x1 conv layer is used)

        num_classes:

            Number of output classes

        activation:

            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

        voxel_size:

            Size of a voxel in the input data, in physical units.

        padding:

            'valid' or 'same', controls the padding on the convolution

        make_iso:

            For anisotropic 3d data, don't downsample z in the beginning,
            until voxel_size is roughly isotropic

        is_training:

            A boolean or placeholder tensor indicating whether or not the
            network is training. Will use dropout and batch norm when
            this is true, but not when false.

        use_batchnorm:

            Whether to use batch norm layers after convolution

        use_conv4d:

            Whether to interpret the input channels as temporal data
            and use conv4d

        global_pool:

            Flag indicating whether or not to do global average pooling.

    '''
    logger.info("Creating VGG-Net")
    num_var_start = get_number_of_tf_variables()

    fov = (1, 1, 1)
    voxel_size = np.array(voxel_size[-3:])
    net = fmaps_in

    if use_conv4d:
        net = tf.expand_dims(net, 1)
    for i, kernel_size in enumerate(kernel_sizes):
        logger.info("%s %s %s %s %s", net, kernel_size, num_fmaps,
                    downsample_factors[i],
                    fmap_inc_factors[i])
        shape = net.get_shape().as_list()
        num_fmaps_in = shape[1]
        if padding.lower() == 'same' and len(shape) == 6 and \
           merge_time_voxel_size is not None and \
           voxel_size[-1] >= merge_time_voxel_size:
            net, fov = conv(
                net, num_fmaps_in, [3, 1, 1, 1],
                activation=activation,
                padding="valid",
                strides=1,
                name='conv_%i_remove_temp' % i)

        net, fov = conv_pass(net,
                             kernel_sizes=kernel_size,
                             num_fmaps=num_fmaps,
                             activation=activation,
                             padding=padding,
                             is_training=is_training,
                             use_batchnorm=use_batchnorm,
                             name='conv_%i' % i,
                             fov=fov,
                             voxel_size=voxel_size)

        num_fmaps *= fmap_inc_factors[i]

        shape = net.get_shape().as_list()
        factors = downsample_factors[i]
        if make_iso and isinstance(factors, int) \
           and factors > 1 and shape[-3] * 2 <= shape[-1]:
            factors = [factors] * (len(shape)-2)
            factors[0] = 1

        net, voxel_size = downsample(
            net,
            factors=factors,
            padding=padding,
            name='pool_%i' % i,
            voxel_size=voxel_size)
        logger.info("current voxel size: %s", voxel_size)

    logger.info("%s", net)
    num_var = get_number_of_tf_variables()
    num_var_conv = num_var - num_var_start

    net, fov = conv_pass(
        net, [1], fc_size,
        activation=activation,
        padding=padding,
        is_training=is_training,
        use_batchnorm=use_batchnorm,
        name='conv_fc1',
        fov=fov, voxel_size=voxel_size)

    if dropout:
        net = tf.layers.dropout(net, rate=dropout,
                                training=is_training)
        logger.info("%s", net)

    if global_pool:
        net = global_average_pool(net)
        logger.info("%s", net)
    else:
        net, fov = conv_pass(
            net, [1], fc_size,
            activation=activation,
            padding=padding,
            is_training=is_training,
            use_batchnorm=use_batchnorm,
            name='conv_fc2',
            fov=fov, voxel_size=voxel_size)

        if dropout:
            net = tf.layers.dropout(net, rate=dropout,
                                    training=is_training)
            logger.info("%s", net)

    net, fov = conv(net, num_classes, 1,
                    activation=None, padding=padding,
                    name='out',
                    fov=fov, voxel_size=voxel_size)

    net = tf.reshape(net, shape=(tf.shape(net)[0], num_classes))
    logger.info("%s", net)

    num_var_end = get_number_of_tf_variables()
    num_var_fc = num_var_end - num_var
    num_var_total = num_var_end - num_var_start
    logger.info('number of variables added (conv part): %i, '
                'number of variables added (fc part): %i, '
                'number of variables added (total): %i, '
                'new total: %i',
                num_var_conv, num_var_fc, num_var_total, num_var_end)
    logger.info("final field of view: %s", fov)
    logger.info("final voxel size: %s", voxel_size)

    summaries = add_summaries()

    net = tf.identity(net, name="logits")
    return net, summaries
