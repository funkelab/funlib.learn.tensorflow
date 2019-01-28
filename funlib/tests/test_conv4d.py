from funlib.learn.tensorflow.models import conv4d
import numpy as np
import tensorflow as tf
import unittest


class TestConv4D(unittest.TestCase):

    def test_conv4d(self):

        i = np.round(np.random.random((1, 1, 10, 11, 12, 13))*100)
        inputs = tf.constant(i, dtype=tf.float32)
        bias_init = tf.constant_initializer(0)

        output = conv4d(
            inputs,
            1,
            (3, 3, 3, 3),
            data_format='channels_first',
            bias_initializer=bias_init,
            name='conv4d_valid')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            o = s.run(output)

            k0 = tf.get_default_graph().get_tensor_by_name(
                'conv4d_valid_3dchan0/kernel:0').eval().flatten()
            k1 = tf.get_default_graph().get_tensor_by_name(
                'conv4d_valid_3dchan1/kernel:0').eval().flatten()
            k2 = tf.get_default_graph().get_tensor_by_name(
                'conv4d_valid_3dchan2/kernel:0').eval().flatten()

            i0 = i[0, 0, 0, 0:3, 0:3, 0:3].flatten()
            i1 = i[0, 0, 1, 0:3, 0:3, 0:3].flatten()
            i2 = i[0, 0, 2, 0:3, 0:3, 0:3].flatten()
            compare = (i0*k0 + i1*k1 + i2*k2).sum()

            np.testing.assert_approx_equal(o[0, 0, 0, 0, 0, 0], compare, 5)

            i0 = i[0, 0, 4, 4:7, 4:7, 4:7].flatten()
            i1 = i[0, 0, 5, 4:7, 4:7, 4:7].flatten()
            i2 = i[0, 0, 6, 4:7, 4:7, 4:7].flatten()
            compare = (i0*k0 + i1*k1 + i2*k2).sum()

            np.testing.assert_approx_equal(o[0, 0, 4, 4, 4, 4], compare, 5)

        output = conv4d(
            inputs,
            1,
            (3, 3, 3, 3),
            data_format='channels_first',
            padding='same',
            kernel_initializer=tf.constant_initializer(1),
            bias_initializer=bias_init,
            name='conv4d_same')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            o = s.run(output)

            i0 = i[0, 0, 0:2, 0:2, 0:2, 0:2]

            np.testing.assert_approx_equal(o[0, 0, 0, 0, 0, 0], i0.sum(), 5)

            i5 = i[0, 0, 4:7, 4:7, 4:7, 4:7]

            np.testing.assert_approx_equal(o[0, 0, 5, 5, 5, 5], i5.sum(), 5)

            i9 = i[0, 0, 8:, 9:, 10:, 11:]

            np.testing.assert_approx_equal(o[0, 0, 9, 10, 11, 12], i9.sum(), 5)
