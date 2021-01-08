# based on https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/efficientnet.py
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
"""EfficientNet models for tf1.

Reference paper:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
        (https://arxiv.org/abs/1905.11946) (ICML 2019)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

import numpy as np
import tensorflow as tf

from .layers import (conv,)
from .DepthwiseConv3D import depthwise_conv_3d

from .utils import (add_summaries,
                    get_number_of_tf_variables,
                    global_average_pool)

logger = logging.getLogger(__name__)

DEFAULT_BLOCKS_ARGS = [{
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 32,
        'filters_out': 16,
        'expand_ratio': 1,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.25
}, {
        'kernel_size': 3,
        'repeats': 2,
        'filters_in': 16,
        'filters_out': 24,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
}, {
        'kernel_size': 5,
        'repeats': 2,
        'filters_in': 24,
        'filters_out': 40,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
}, {
        'kernel_size': 3,
        'repeats': 3,
        'filters_in': 40,
        'filters_out': 80,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
}, {
        'kernel_size': 5,
        'repeats': 3,
        'filters_in': 80,
        'filters_out': 112,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.25
}, {
        'kernel_size': 5,
        'repeats': 4,
        'filters_in': 112,
        'filters_out': 192,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.25
# }, {
#         'kernel_size': 3,
#         'repeats': 1,
#         'filters_in': 192,
#         'filters_out': 320,
#         'expand_ratio': 6,
#         'id_skip': True,
#         'strides': 1,
#         'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
                'scale': 2.0,
                'mode': 'fan_out',
                'distribution': 'truncated_normal'
        }
}

DENSE_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
                'scale': 1. / 3.,
                'mode': 'fan_out',
                'distribution': 'uniform'
        }
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
            https://arxiv.org/abs/1905.11946) (ICML 2019)

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    If you have never configured it, it defaults to `"channels_last"`.

    Arguments:
        include_top: Whether to include the fully-connected
                layer at the top of the network. Defaults to True.
        weights: One of `None` (random initialization),
                    'imagenet' (pre-training on ImageNet),
                    or the path to the weights file to be loaded. Defaults to 'imagenet'.
        input_tensor: Optional Keras tensor
                (i.e. output of `layers.Input()`)
                to use as image input for the model.
        input_shape: Optional shape tuple, only to be specified
                if `include_top` is False.
                It should have exactly 3 inputs channels.
        pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`. Defaults to None.
                - `None` means that the output of the model will be
                        the 4D tensor output of the
                        last convolutional layer.
                - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                        be applied.
        classes: Optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified. Defaults to 1000 (number of
                ImageNet classes).
        classifier_activation: A `str` or callable. The activation function to use
                on the "top" layer. Ignored unless `include_top=True`. Set
                `classifier_activation=None` to return the logits of the "top" layer.
                Defaults to 'softmax'.

    Returns:
        A `keras.Model` instance.
"""


def EfficientNet(
        net,
        width_coefficient,
        depth_coefficient,
        default_size,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation='swish',
        blocks_args='default',
        model_name='efficientnet',
        padding='SAME',
        is_training=None,
        make_iso=False,
        num_classes=None,
        voxel_size=None,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

    Reference paper:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
            https://arxiv.org/abs/1905.11946) (ICML 2019)

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Arguments:
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
                layer at the top of the network.
        weights: one of `None` (random initialization),
                    'imagenet' (pre-training on ImageNet),
                    or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
                (i.e. output of `layers.Input()`)
                to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
                if `include_top` is False.
                It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                        the 4D tensor output of the
                        last convolutional layer.
                - `avg` means that global average pooling
                        will be applied to the output of the
                        last convolutional layer, and thus
                        the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                        be applied.
        classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
                on the "top" layer. Ignored unless `include_top=True`. Set
                `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
        A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    if blocks_args == 'default':
        blocks_args = DEFAULT_BLOCKS_ARGS


    # Determine proper input shape

    input_shape = net.get_shape().as_list()

    bn_axis = 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    fov = (1, 1, 1)
    voxel_size = np.array(voxel_size[-3:])

    # Build stem
    # net, fov = conv(net, round_filters(32), 3,
    #                 activation=None,
    #                 padding=padding,
    #                 strides=2,
    #                 use_bias=False,
    #                 name='conv1',
    #                 fov=fov, voxel_size=voxel_size)
    net, fov = conv(net, round_filters(32), 3,
                    activation=None,
                    padding=padding,
                    strides=1,
                    use_bias=False,
                    name='conv1',
                    fov=fov, voxel_size=voxel_size)
    logger.info("%s", net)
    # x = layers.Conv2D(
    #         round_filters(32),
    #         3,
    #         strides=2,
    #         padding='valid',
    #         use_bias=False,
    #         kernel_initializer=CONV_KERNEL_INITIALIZER,
    #         name='stem_conv')(x)

    net = tf.layers.batch_normalization(
            net,
            axis=bn_axis,
            training=is_training,
            epsilon=1.0001e-5,
            name='in_bn')
    logger.info("%s", net)
    net = tf.keras.activations.get(activation)(net)
    logger.info("%s", net)
    # x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    # x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            net, fov, voxel_size = block(
                net,
                activation,
                drop_connect_rate * b / blocks,
                name='block{}{}_'.format(i + 1, chr(j + 97)),
                padding=padding,
                voxel_size=voxel_size,
                fov=fov,
                is_training=is_training,
                make_iso=make_iso,
                **args)
            b += 1

    # Build top
    net, fov = conv(net, round_repeats(1280), 1,
                    activation=None,
                    padding=padding,
                    use_bias=False,
                    name='conv_end',
                    fov=fov, voxel_size=voxel_size)
    # x = layers.Conv2D(
    #         round_filters(1280),
    #         1,
    #         padding='same',
    #         use_bias=False,
    #         kernel_initializer=CONV_KERNEL_INITIALIZER,
    #         name='top_conv')(x)
    net = tf.layers.batch_normalization(
            net,
            axis=bn_axis,
            training=is_training,
            epsilon=1.0001e-5,
            name='out_bn')
    logger.info("%s", net)
    net = tf.keras.activations.get(activation)(net)
    logger.info("%s", net)

    # x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    # x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
        net = global_average_pool(net, keep_dims=False)
        # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            # out_shape = tuple(net.get_shape().as_list())
            # shape = net.get_shape().as_list()
            # shape[2:] = [1] * (len(out_shape)-2)
            net = tf.layers.dropout(
                net, rate=dropout_rate, training=is_training)
            logger.info("%s", net)
            # x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        # imagenet_utils.validate_activation(classifier_activation, weights)
        net = tf.layers.dense(net, num_classes, name="dense_out")
        # x = layers.Dense(
        #         classes,
        #         activation=classifier_activation,
        #         kernel_initializer=DENSE_KERNEL_INITIALIZER,
        #         name='predictions')(x)
    else:
        net = global_average_pool(net, keep_dims=True)
        logger.info("%s", net)
        net, fov = conv(net, num_classes, 1,
                    activation=None, padding=padding,
                    name='out',
                    fov=fov, voxel_size=voxel_size)
        logger.info("%s", net)
        net = tf.reshape(net, shape=(tf.shape(net)[0], num_classes))

        # if pooling == 'avg':
        #     x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        # elif pooling == 'max':
        #     x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    logger.info("%s", net)
    
    # # Ensure that the model takes into account
    # # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = layer_utils.get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input

    # # Create model.
    # model = training.Model(inputs, x, name=model_name)

    # # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         file_suffix = '.h5'
    #         file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
    #     else:
    #         file_suffix = '_notop.h5'
    #         file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
    #     file_name = model_name + file_suffix
    #     weights_path = data_utils.get_file(
    #             file_name,
    #             BASE_WEIGHTS_PATH + file_name,
    #             cache_subdir='models',
    #             file_hash=file_hash)
    #     model.load_weights(weights_path)
    # elif weights is not None:
    #     model.load_weights(weights)

    logger.info("field of view: %s", fov)
    logger.info("final voxel size: %s", voxel_size)

    summaries = add_summaries()
    return net, summaries


def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_in=32,
          filters_out=16,
          kernel_size=3,
          padding="SAME",
          is_training=None,
          voxel_size=None,
          make_iso=False,
          fov=None,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True):
    """An inverted residual block.

    Arguments:
            inputs: input tensor.
            activation: activation function.
            drop_rate: float between 0 and 1, fraction of the input units to drop.
            name: string, block label.
            filters_in: integer, the number of input filters.
            filters_out: integer, the number of output filters.
            kernel_size: integer, the dimension of the convolution window.
            strides: integer, the stride of the convolution.
            expand_ratio: integer, scaling coefficient for the input filters.
            se_ratio: float between 0 and 1, fraction to squeeze the input filters.
            id_skip: boolean.

    Returns:
            output tensor for the block.
    """
    bn_axis = 1

    net = inputs

    shape = net.get_shape().as_list()
    if isinstance(strides, int):
        strides = [strides]*(len(shape) - 2)

    if make_iso and len(shape) > 4 and \
       strides[-1] > 1 and shape[-3] * 2 <= shape[-1]:
        strides[-3] = 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        net, fov = conv(net, filters, 1,
                        activation=None,
                        padding=padding,
                        use_bias=False,
                        # strides=2,
                        name=name+'expand_conv',
                        fov=fov, voxel_size=voxel_size)
        # x = layers.Conv2D(
        #         filters,
        #         1,
        #         padding='same',
        #         use_bias=False,
        #         kernel_initializer=CONV_KERNEL_INITIALIZER,
        #         name=name + 'expand_conv')(
        #                 inputs)
        net = tf.layers.batch_normalization(
            net,
            axis=bn_axis,
            training=is_training,
            epsilon=1.0001e-5,
            name=name+'expand_bn')
        logger.info("%s", net)
        net = tf.keras.activations.get(activation)(net)
        logger.info("%s", net)
        # x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        # x = layers.Activation(activation, name=name + 'expand_activation')(x)
    # else:
    #     x = inputs

    # Depthwise Convolution
    # if strides == 2:
    #     x = layers.ZeroPadding2D(
    #             padding=imagenet_utils.correct_pad(x, kernel_size),
    #             name=name + 'dwconv_pad')(x)
    #     conv_pad = 'valid'
    # else:
    #     conv_pad = 'same'
    logger.info("%s", net)
    # net, fov = conv(net, filters, kernel_size,
    #                     activation=None,
    #                     padding=padding,
    #                     use_bias=False,
    #                     strides=strides,
    #                     name=name+'fake_depth_conv',
    #                     fov=fov, voxel_size=voxel_size)

    net = depthwise_conv_3d(
        net,
        kernel_size,
        strides=strides,
        padding=padding,
        activation=None,
        data_format="channels_first",
        depth_multiplier=1,
        use_bias=False)

    logger.info("%s", net)
    fov = tuple(
        f + (kernel_size - 1)*vs
        for f, vs
        in zip(fov, voxel_size)
    )

    # x = layers.DepthwiseConv2D(
    #         kernel_size,
    #         strides=strides,
    #         padding=conv_pad,
    #         use_bias=False,
    #         depthwise_initializer=CONV_KERNEL_INITIALIZER,
    #         name=name + 'dwconv')(x)
    net = tf.layers.batch_normalization(
            net,
            axis=bn_axis,
            training=is_training,
            epsilon=1.0001e-5,
            name=name+'_bn')
    logger.info("%s", net)
    net = tf.keras.activations.get(activation)(net)
    logger.info("%s", net)

    # x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    # x = layers.Activation(activation, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))

        net_se = global_average_pool(net)
        # net_se = tf.reshape(net_se, shape=(1, 1, 1, filters))
        # se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        # se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        net_se, fov = conv(net_se, filters_se, 1,
                    activation=activation,
                    padding=padding,
                    # strides=2,
                    name=name+'se_reduce',
                    fov=fov, voxel_size=voxel_size)

        # se = layers.Conv2D(
        #         filters_se,
        #         1,
        #         padding='same',
        #         activation=activation,
        #         kernel_initializer=CONV_KERNEL_INITIALIZER,
        #         name=name + 'se_reduce')(
        #                 se)
        net_se, fov = conv(net_se, filters, 1,
                    activation="sigmoid",
                    padding=padding,
                    # strides=2,
                    name=name+'se_expand',
                    fov=fov, voxel_size=voxel_size)

        # se = layers.Conv2D(
        #         filters,
        #         1,
        #         padding='same',
        #         activation='sigmoid',
        #         kernel_initializer=CONV_KERNEL_INITIALIZER,
        #         name=name + 'se_expand')(se)
        # x = layers.multiply([x, se], name=name + 'se_excite')
        logger.info("%s", (filters_in, filters, expand_ratio, net, net_se))
        net = tf.keras.layers.multiply([net, net_se], name=name + 'se_excite')

    # Output phase
    net, fov = conv(net, filters_out, 1,
                    activation=None,
                    padding=padding,
                    # strides=2,
                    use_bias=False,
                    name=name+'project_conv',
                    fov=fov, voxel_size=voxel_size)

    # x = layers.Conv2D(
    #         filters_out,
    #         1,
    #         padding='same',
    #         use_bias=False,
    #         kernel_initializer=CONV_KERNEL_INITIALIZER,
    #         name=name + 'project_conv')(x)
    net = tf.layers.batch_normalization(
            net,
            axis=bn_axis,
            training=is_training,
            epsilon=1.0001e-5,
            name=name+'out_bn')
    logger.info("%s", net)
    # net = tf.keras.activations.get(activation)(net)
    # logger.info("%s", net)

    # x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            out_shape = tuple(net.get_shape().as_list())
            shape = net.get_shape().as_list()
            shape[2:] = [1] * (len(out_shape)-2)
            net = tf.layers.dropout(
                net, rate=drop_rate, noise_shape=shape, training=is_training)
            logger.info("%s", net)

        # if drop_rate > 0:
        #     x = layers.Dropout(
        #             drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
        # x = layers.add([x, inputs], name=name + 'add')
        net = tf.keras.layers.add([net, inputs], name=name + 'add')
    return net, fov, voxel_size


def EfficientNetBX(net, **kwargs):
    efficientnet_size = kwargs['efficientnet_size']
    del kwargs['efficientnet_size']
    if efficientnet_size == "B0":
        net = EfficientNetB0(net, **kwargs)
    elif efficientnet_size == "B1":
        net = EfficientNetB1(net, **kwargs)
    elif efficientnet_size == "B2":
        net = EfficientNetB2(net, **kwargs)
    elif efficientnet_size == "B3":
        net = EfficientNetB3(net, **kwargs)
    elif efficientnet_size == "B4":
        net = EfficientNetB4(net, **kwargs)
    elif efficientnet_size == "B5":
        net = EfficientNetB5(net, **kwargs)
    elif efficientnet_size == "B6":
        net = EfficientNetB6(net, **kwargs)
    elif efficientnet_size == "B7":
        net = EfficientNetB7(net, **kwargs)
    elif efficientnet_size == "B8":
        net = EfficientNetB8(net, **kwargs)
    elif efficientnet_size == "B9":
        net = EfficientNetB9(net, **kwargs)
    elif efficientnet_size == "B10":
        net = EfficientNetB10(net, **kwargs)
    else:
        raise RuntimeError("invalid efficientnet_size %s" % efficientnet_size)

    return net

def EfficientNetB0(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.0,
        1.0,
        224,
        0.2,
        model_name='efficientnetb0',
        **kwargs)

def EfficientNetB1(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.0,
        1.1,
        240,
        0.2,
        model_name='efficientnetb1',
        **kwargs)

def EfficientNetB2(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.1,
        1.2,
        260,
        0.3,
        model_name='efficientnetb2',
        **kwargs)

def EfficientNetB3(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.2,
        1.4,
        300,
        0.3,
        model_name='efficientnetb3',
        **kwargs)

def EfficientNetB4(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.4,
        1.8,
        380,
        0.4,
        model_name='efficientnetb4',
        **kwargs)

def EfficientNetB5(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.6,
        2.2,
        456,
        0.4,
        model_name='efficientnetb5',
        **kwargs)

def EfficientNetB6(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.8,
        2.6,
        528,
        0.5,
        model_name='efficientnetb6',
        **kwargs)

def EfficientNetB7(net,
                   **kwargs):
    return EfficientNet(
        net,
        2.0,
        3.1,
        600,
        0.5,
        model_name='efficientnetb7',
        **kwargs)


def EfficientNetB8(net,
                   **kwargs):
    return EfficientNet(
        net,
        1.0,
        1.0,
        600,
        0.5,
        model_name='efficientnetb8',
        **kwargs)


def EfficientNetB9(net,
                   **kwargs):
    return EfficientNet(
        net,
        0.5,
        0.5,
        600,
        0.2,
        model_name='efficientnetb9',
        **kwargs)


def EfficientNetB10(net,
                   **kwargs):
    return EfficientNet(
        net,
        0.5,
        0.5,
        600,
        0.5,
        model_name='efficientnetb10',
        **kwargs)
