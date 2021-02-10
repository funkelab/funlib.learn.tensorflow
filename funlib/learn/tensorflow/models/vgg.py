import logging

import tensorflow as tf

from funlib.learn.tensorflow.models.unet import downsample, conv_pass
from .utils import get_number_of_tf_variables

logger = logging.getLogger(__name__)


def global_average_pool(net):
    return tf.reduce_mean(net, [2, 3, 4], keep_dims=True,
                          name='global_avg_pool')


def add_summaries(net, sums):
    sums.append(tf.summary.histogram(net.op.name, net))
    name = net.op.name.split("/")[0]
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             name)
    sums.append(tf.summary.histogram(name + '/kernel', vars[0]))
    sums.append(tf.summary.histogram(name + '/bias', vars[1]))


def vgg(f_in,
        *,  # this indicates that all following arguments are keyword arguments
        kernel_sizes,
        num_fmaps,
        fmap_inc_factors,
        downsample_factors,
        fc_size,
        num_classes,
        is_training,
        activation='relu',
        voxel_size=(1, 1, 1),
        padding='same',
        batch_norm=True,
        global_pool=True):

    ''' Create a VGG-Net:

    Args:
        f_in:

            The input tensor of shape ``(batch_size, channels, [length,] depth,
            height, width)``.

        kernel_sizes:

            Sizes of the kernels to use. Forwarded to conv_pass (see u-net).

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

        num_classes:

            Number of output classes

        is_training:

            A boolean or placeholder tensor indicating whether or not the
            network is training. Will use dropout and batch norm when
            this is true, but not when false.

        activation:

            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

        voxel_size:

            Size of a voxel in the input data, in physical units.

        padding:

            'valid' or 'same', controls the padding on the convolution

        batch_norm:

            Flag indicating whether or not to do batch normalization.
            Default is true.

        global_pool:

            Flag indicating whether or not to do global average pooling.
            Default is true.

    '''
    logger.info("Creating VGG-Net")
    num_var_start = get_number_of_tf_variables()

    fov = (1, 1, 1)
    net = f_in
    sums = []
    for i, kernel_size in enumerate(kernel_sizes):
        logger.info("%s %s %s %s %s", net, kernel_size, num_fmaps,
                    downsample_factors[i],
                    fmap_inc_factors[i])
        net, fov = conv_pass(net,
                             kernel_sizes=kernel_size,
                             num_fmaps=num_fmaps,
                             padding=padding,
                             activation=activation,
                             name='conv_%i' % i,
                             fov=fov,
                             voxel_size=voxel_size)
        add_summaries(net, sums)

        num_fmaps *= fmap_inc_factors[i]
        if batch_norm:
            net = tf.layers.batch_normalization(net, training=is_training)

        net, voxel_size = downsample(
                net,
                downsample_factors[i],
                'pool_%i' % i,
                voxel_size=voxel_size)

    logger.info(net)
    num_var_conv = get_number_of_tf_variables()
    var_added = num_var_conv - num_var_start
    logger.info('number of variables added (conv part): %i, '
                'new total: %i', var_added, num_var_conv)

    net, fov = conv_pass(
            net, [1], fc_size, padding=padding,
            name='conv_fc7', fov=fov, voxel_size=voxel_size)
    add_summaries(net, sums)

    logger.info(net)
    if global_pool:
        net = global_average_pool(net)
    logger.info(net)
    net = tf.layers.dropout(net, training=is_training)
    net, fov = conv_pass(net, [1], num_classes,
                         padding=padding, name='conv_fc8',
                         activation=None, fov=fov, voxel_size=voxel_size)
    add_summaries(net, sums)

    logger.info(net)
    net = tf.reshape(net, shape=(tf.shape(net)[0], num_classes))
    logger.info(net)

    num_var_end = get_number_of_tf_variables()
    var_added = num_var_end - num_var_conv
    var_added_total = num_var_end - num_var_start
    logger.info('number of variables added (fc part): %i, '
                'number of variables added (total): %i, '
                'new total: %i', var_added, var_added_total, num_var_end)

    return net, sums
