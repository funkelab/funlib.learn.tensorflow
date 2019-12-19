import logging

import tensorflow as tf

from funlib.learn.tensorflow.models.unet import downsample, conv_pass

logger = logging.getLogger(__name__)


def get_number_of_tf_variables():
    '''Returns number of trainable variables in tensorflow graph collection'''
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


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


def vgg(fmaps_in,
        is_training,
        voxel_size,
        kernel_sizes,
        downsample_factors,
        fmap_inc_factors,
        num_fmaps,
        fc_size,
        num_classes,
        padding='same',
        activation='relu',
        batch_norm=True,
        global_pool=True):
    ''' Create a VGG-Net:
    '''
    logger.info("Creating VGG-Net")
    num_var_start = get_number_of_tf_variables()

    fov = (1, 1, 1)
    voxel_size = voxel_size
    num_fmaps = num_fmaps
    net = fmaps_in
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
