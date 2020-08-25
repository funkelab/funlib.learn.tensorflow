import math

import tensorflow as tf


def add_summaries():
    summaries = []

    # idea: get histogram of activations, not working yet
    # ops = tf.get_default_graph().get_operations()
    # for o in ops:
    #         summaries.append(tf.summary.histogram(o.name, o))

    vars = tf.trainable_variables()
    for v in vars:
        summaries.append(
            tf.summary.histogram(v.name.replace(":", "_"), v))

    return summaries


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


def crop(a, shape):
    '''Crop a to a new shape, centered in a.

    Args:

        a:

            The input tensor.

        shape:

            A list (not a tensor) with the requested shape.
    '''

    # l, d, h, w
    in_shape = a.get_shape().as_list()

    offset = list([
        (i - s)//2
        for i, s in zip(in_shape, shape)
    ])

    b = tf.slice(a, offset, shape)

    return b


def crop_spatial_temporal(fmaps_in, shape):
    '''Crop spatial and time dimensions to match shape.

    Args:

        fmaps_in:

            The input tensor of shape ``(b, c, z, y, x)`` (for 3D) or ``(b, c,
            t, z, y, x)`` (for 4D).

        shape:

            A list (not a tensor) with the requested shape ``[_, _, z, y, x]``
            (for 3D) or ``[_, _, t, z, y, x]`` (for 4D).
    '''

    in_shape = fmaps_in.get_shape().as_list()

    # Explicitly handle number of dimensions
    in_is_4d = len(in_shape) == 6
    in_is_2d = len(in_shape) == 4
    out_is_4d = len(shape) == 6

    if in_is_4d and not out_is_4d:
        # set output shape for time to 1
        shape = shape[0:2] + [1] + shape[2:]

    if in_is_4d:
        offset = [
            0,  # batch
            0,  # channel
            (in_shape[2] - shape[2])//2,  # t
            (in_shape[3] - shape[3])//2,  # z
            (in_shape[4] - shape[4])//2,  # y
            (in_shape[5] - shape[5])//2,  # x
        ]
        size = [
            in_shape[0],
            in_shape[1],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
        ]
    elif in_is_2d:
        offset = [
            0,  # batch
            0,  # channel
            (in_shape[2] - shape[2])//2,  # y
            (in_shape[3] - shape[3])//2,  # x
        ]
        size = [
            in_shape[0],
            in_shape[1],
            shape[2],
            shape[3],
        ]

    else:
        offset = [
            0,  # batch
            0,  # channel
            (in_shape[2] - shape[2])//2,  # z
            (in_shape[3] - shape[3])//2,  # y
            (in_shape[4] - shape[4])//2,  # x
        ]
        size = [
            in_shape[0],
            in_shape[1],
            shape[2],
            shape[3],
            shape[4],
        ]

    fmaps = tf.slice(fmaps_in, offset, size)

    if in_is_4d and not out_is_4d:
        # remove time dimension
        shape = shape[0:2] + shape[3:]
        fmaps = tf.reshape(fmaps, shape)

    return fmaps


def crop_to_factor(fmaps_in, factor, kernel_sizes):
    '''Crop feature maps to ensure translation equivariance with stride of
    upsampling factor. This should be done right after upsampling, before
    application of the convolutions with the given kernel sizes.

    The crop could be done after the convolutions, but it is more efficient to
    do that before (feature maps will be smaller).
    '''

    shape = fmaps_in.get_shape().as_list()
    spatial_dims = len(shape) - 2
    spatial_shape = shape[-spatial_dims:]

    # the crop that will already be done due to the convolutions
    convolution_crop = list(
        sum(
            (ks if isinstance(ks, int) else ks[d]) - 1
            for ks in kernel_sizes
        )
        for d in range(spatial_dims)
    )
    print("crop_to_factor: factor =", factor)
    print("crop_to_factor: kernel_sizes =", kernel_sizes)
    print("crop_to_factor: convolution_crop =", convolution_crop)

    # we need (spatial_shape - convolution_crop) to be a multiple of factor,
    # i.e.:
    #
    # (s - c) = n*k
    #
    # we want to find the largest n for which s' = n*k + c <= s
    #
    # n = floor((s - c)/k)
    #
    # this gives us the target shape s'
    #
    # s' = n*k + c

    ns = (
        int(math.floor(float(s - c)/f))
        for s, c, f in zip(spatial_shape, convolution_crop, factor)
    )
    target_spatial_shape = tuple(
        n*f + c
        for n, c, f in zip(ns, convolution_crop, factor)
    )

    if target_spatial_shape != tuple(spatial_shape):

        assert all((
                (t > c) for t, c in zip(
                    target_spatial_shape,
                    convolution_crop))
            ), \
            "Feature map with shape %s is too small to ensure translation " \
            "equivariance with factor %s and following convolutions %s" % (
                shape,
                factor,
                kernel_sizes)

        target_shape = list(shape)
        target_shape[-spatial_dims:] = target_spatial_shape

        print("crop_to_factor: shape =", shape)
        print("crop_to_factor: spatial_shape =", spatial_shape)
        print("crop_to_factor: target_spatial_shape =", target_spatial_shape)
        print("crop_to_factor: target_shape =", target_shape)
        fmaps = crop_spatial_temporal(
            fmaps_in,
            target_shape)
    else:
        fmaps = fmaps_in

    return fmaps
