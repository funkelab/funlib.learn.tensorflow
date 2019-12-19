from .conv4d import conv4d
import math
import numpy as np
import tensorflow as tf


def conv_pass(
        fmaps_in,
        kernel_sizes,
        num_fmaps,
        activation='relu',
        name='conv_pass',
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1),
        padding='valid'):
    '''Create a convolution pass::

        f_in --> f_1 --> ... --> f_n

    where each ``-->`` is a convolution followed by a (non-linear) activation
    function. One convolution will be performed for each entry in
    ``kernel_sizes``. Each convolution will decrease the size of the feature
    maps by ``kernel_size-1``.

    Args:

        f_in:

            The input tensor of shape ``(batch_size, channels, [length,] depth,
            height, width)``.

        kernel_sizes:

            Sizes of the kernels to use. Forwarded to tf.layers.conv3d.

        num_fmaps:

            The number of feature maps to produce with each convolution.

        activation:

            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

        name:

            Base name for the conv layer.

        fov:

            Field of view of fmaps_in, in physical units.

        voxel_size:

            Size of a voxel in the input data, in physical units.

        padding:

            'valid' or 'same', controls the padding on the convolution

    Returns:

        (fmaps, fov):

            The feature maps after the last convolution, and a tuple
            representing the field of view.
    '''

    fmaps = fmaps_in
    if activation is not None:
        activation = getattr(tf.nn, activation)

    for i, kernel_size in enumerate(kernel_sizes):
        in_shape = tuple(fmaps.get_shape().as_list())

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
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=kernel_size,
            padding=padding,
            data_format='channels_first',
            activation=activation,
            name=name + '_%i' % i)

        out_shape = tuple(fmaps.get_shape().as_list())

        # eliminate t dimension if length is 1
        if len(out_shape) == 6:
            length = out_shape[2]
            if length == 1:
                out_shape = out_shape[0:2] + out_shape[3:]
                fmaps = tf.reshape(fmaps, out_shape)

    return fmaps, fov


def downsample(
        fmaps_in,
        factors,
        name='down',
        voxel_size=(1, 1, 1)):
    voxel_size = tuple(vs*fac for vs, fac in zip(voxel_size, factors))
    in_shape = fmaps_in.get_shape().as_list()

    # Explicitly handle number of dimensions
    is_4d = len(in_shape) == 6
    is_2d = len(in_shape) == 4

    if is_4d:
        orig_in_shape = in_shape
        # store time dimension in channels
        fmaps_in = tf.reshape(fmaps_in, (
            in_shape[0],
            in_shape[1]*in_shape[2],
            in_shape[3],
            in_shape[4],
            in_shape[5]))
        in_shape = fmaps_in.get_shape().as_list()

    if not np.all(np.array(in_shape[2:]) % np.array(factors) == 0):
        raise RuntimeWarning(
            "Input shape %s is not evenly divisible by downsample factor %s." %
            (in_shape[2:], factors))

    if is_2d:
        fmaps = tf.layers.max_pooling2d(
            fmaps_in,
            pool_size=factors,
            strides=factors,
            padding='valid',
            data_format='channels_first',
            name=name,
        )
    else:
        fmaps = tf.layers.max_pooling3d(
            fmaps_in,
            pool_size=factors,
            strides=factors,
            padding='valid',
            data_format='channels_first',
            name=name)

    if is_4d:

        out_shape = fmaps.get_shape().as_list()

        # restore time dimension
        fmaps = tf.reshape(fmaps, (
            orig_in_shape[0],
            orig_in_shape[1],
            orig_in_shape[2],
            out_shape[2],
            out_shape[3],
            out_shape[4]))

    return fmaps, voxel_size


def repeat(
        fmaps_in,
        multiples):

    expanded = tf.expand_dims(fmaps_in, -1)
    tiled = tf.tile(expanded, multiples=(1,) + multiples)
    repeated = tf.reshape(tiled, tf.shape(fmaps_in) * multiples)

    return repeated


def upsample(
        fmaps_in,
        factors,
        num_fmaps,
        activation='relu',
        name='up',
        voxel_size=(1, 1, 1),
        constant_upsample=False):
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

        activation (``string``):

            Which activation function to use.

        name (``string``):

            Name of the operator.

        voxel_size (``tuple`` of ``int``, optional):

            Voxel size of the input feature maps. Used to compute the voxel
            size of the output.

        constant_upsample (``bool``, optional):

            Whether to restrict the transpose convolution kernels to be
            constant values. This might help to reduce checker board artifacts.

    Returns:

        `(fmaps, voxel_size)`, with `fmaps` of shape `(b, num_fmaps, d*f_z,
        h*f_y, w*f_x)`.
    '''

    # Explicitly handle number of dimensions
    is_2d = len(fmaps_in.get_shape().as_list()) == 4

    voxel_size = tuple(vs/fac for vs, fac in zip(voxel_size, factors))
    if activation is not None:
        activation = getattr(tf.nn, activation)

    if constant_upsample:

        in_shape = tuple(fmaps_in.get_shape().as_list())
        num_fmaps_in = in_shape[1]
        num_fmaps_out = num_fmaps
        out_shape = (
            in_shape[0],
            num_fmaps_out) + tuple(s*f for s, f in zip(in_shape[2:], factors))

        if is_2d:
            # (num_fmaps_out * num_fmaps_in)
            kernel_variables = tf.get_variable(
                name + '_kernel_variables',
                (num_fmaps_out * num_fmaps_in,),
                dtype=tf.float32)
            # (1, 1, num_fmaps_out, num_fmaps_in)
            kernel_variables = tf.reshape(
                kernel_variables,
                (1, 1) + (num_fmaps_out, num_fmaps_in))
            # (f_y, f_x, num_fmaps_out, num_fmaps_in)
            constant_upsample_filter = repeat(
                kernel_variables,
                tuple(factors) + (1, 1))

            fmaps = tf.nn.conv2d_transpose(
                fmaps_in,
                filter=constant_upsample_filter,
                output_shape=out_shape,
                strides=(1, 1) + tuple(factors),
                padding='VALID',
                data_format='NCHW',
                name=name)

            if activation is not None:
                fmaps = activation(fmaps)

        else:
            # (num_fmaps_out * num_fmaps_in)
            kernel_variables = tf.get_variable(
                name + '_kernel_variables',
                (num_fmaps_out * num_fmaps_in,),
                dtype=tf.float32)
            # (1, 1, 1, num_fmaps_out, num_fmaps_in)
            kernel_variables = tf.reshape(
                kernel_variables,
                (1, 1, 1) + (num_fmaps_out, num_fmaps_in))
            # (f_z, f_y, f_x, num_fmaps_out, num_fmaps_in)
            constant_upsample_filter = repeat(
                kernel_variables,
                tuple(factors) + (1, 1))

            fmaps = tf.nn.conv3d_transpose(
                fmaps_in,
                filter=constant_upsample_filter,
                output_shape=out_shape,
                strides=(1, 1) + tuple(factors),
                padding='VALID',
                data_format='NCDHW',
                name=name)

            if activation is not None:
                fmaps = activation(fmaps)

    else:
        if is_2d:
            fmaps = tf.layers.conv2d_transpose(
                fmaps_in,
                filters=num_fmaps,
                kernel_size=factors,
                strides=factors,
                padding='valid',
                data_format='channels_first',
                activation=activation,
                name=name,
                )

        else:
            fmaps = tf.layers.conv3d_transpose(
                fmaps_in,
                filters=num_fmaps,
                kernel_size=factors,
                strides=factors,
                padding='valid',
                data_format='channels_first',
                activation=activation,
                name=name)

    return fmaps, voxel_size


def crop(fmaps_in, shape):
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

    if target_spatial_shape != spatial_shape:

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
        fmaps = crop(
            fmaps_in,
            target_shape)
    else:
        fmaps = fmaps_in

    return fmaps


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


def unet(
        fmaps_in,
        num_fmaps,
        fmap_inc_factors,
        downsample_factors,
        kernel_size_down=None,
        kernel_size_up=None,
        activation='relu',
        layer=0,
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1),
        num_fmaps_out=None,
        num_heads=1,
        constant_upsample=False):
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

        fmap_inc_factors:

            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.

        downsample_factors:

            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.

        kernel_size_down (optional):

            List of lists of kernel sizes. The number of sizes in a list
            determines the number of convolutional layers in the corresponding
            level of the build on the left side. Kernel sizes can be given as
            tuples or integer. If not given, each convolutional pass will
            consist of two 3x3x3 convolutions.

        kernel_size_up (optional):

            List of lists of kernel sizes. The number of sizes in a list
            determines the number of convolutional layers in the corresponding
            level of the build on the right side. Within one of the lists going
            from left to right. Kernel sizes can be given as tuples or integer.
            If not given, each convolutional pass will consist of two 3x3x3
            convolutions.

        activation:

            Which activation to use after a convolution. Accepts the name of
            any tensorflow activation function (e.g., ``relu`` for
            ``tf.nn.relu``).

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
    '''
    num_var_start = get_number_of_tf_variables()
    prefix = "    "*layer
    print(prefix + "Creating U-Net layer %i" % layer)
    print(prefix + "f_in: " + str(fmaps_in.shape))
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
        name='unet_layer_%i_left' % layer,
        fov=fov,
        voxel_size=voxel_size)

    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))

    num_var_end = get_number_of_tf_variables()
    var_added = num_var_end - num_var_start
    if bottom_layer:
        print(prefix + "bottom layer")
        print(prefix + "f_out: " + str(f_left.shape))
        if num_heads > 1:
            f_left = [f_left] * num_heads
        print(prefix + 'number of variables added: %i, '
                       'new total: %i' % (var_added, num_var_end))
        return f_left, fov, voxel_size

    # downsample
    g_in, voxel_size = downsample(
        f_left,
        downsample_factors[layer],
        'unet_down_%i_to_%i' % (layer, layer + 1),
        voxel_size=voxel_size)

    print(prefix + 'number of variables added: %i, '
                   'new total: %i' % (var_added, num_var_end))
    # recursive U-net
    g_outs, fov, voxel_size = unet(
        g_in,
        num_fmaps=num_fmaps*fmap_inc_factors[layer],
        fmap_inc_factors=fmap_inc_factors,
        downsample_factors=downsample_factors,
        kernel_size_down=kernel_size_down,
        kernel_size_up=kernel_size_up,
        activation=activation,
        layer=layer+1,
        fov=fov,
        voxel_size=voxel_size,
        num_heads=num_heads,
        constant_upsample=constant_upsample)
    if num_heads == 1:
        g_outs = [g_outs]

    # For Multi-Headed UNet: Create this path multiple times.
    f_outs = []
    for head_num, g_out in enumerate(g_outs):
        num_var_start = get_number_of_tf_variables()
        with tf.variable_scope('decoder_%i_layer_%i' % (head_num, layer)):
            if num_heads > 1:
                print(prefix + 'head number: %i' % head_num)
            print(prefix + "g_out: " + str(g_out.shape))
            # upsample
            g_out_upsampled, voxel_size = upsample(
                g_out,
                downsample_factors[layer],
                num_fmaps,
                activation=activation,
                name='unet_up_%i_to_%i' % (layer + 1, layer),
                voxel_size=voxel_size,
                constant_upsample=constant_upsample)

            print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

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

            print(
                prefix + "g_out_upsampled_cropped: " +
                str(g_out_upsampled.shape))

            # copy-crop
            f_left_cropped = crop(f_left,
                                  g_out_upsampled.get_shape().as_list())

            print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

            # concatenate along channel dimension
            f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

            print(prefix + "f_right: " + str(f_right.shape))

            if layer == 0 and num_fmaps_out is not None:
                num_fmaps = max(num_fmaps_out, num_fmaps)

            # convolve
            f_out, fov = conv_pass(
                f_right,
                kernel_sizes=kernel_size_up[layer],
                num_fmaps=num_fmaps,
                name='unet_layer_%i_right' % (layer),
                fov=fov,
                voxel_size=voxel_size)

            print(prefix + "f_out: " + str(f_out.shape))
            f_outs.append(f_out)
            num_var_end = get_number_of_tf_variables()
            var_added = num_var_end - num_var_start
            print(prefix + 'number of variables added: %i, '
                           'new total: %i' % (var_added, num_var_end))
    if num_heads == 1:
        f_outs = f_outs[0]  # Backwards compatibility.
    return f_outs, fov, voxel_size
