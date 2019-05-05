from funlib.learn.tensorflow.models import upsample, repeat
import numpy as np
import tensorflow as tf
import unittest


def assert_all_equal(a):

    np.testing.assert_array_equal(a, a.flatten()[0])


class TestConv4D(unittest.TestCase):

    def test_repeat(self):

        i = np.round(np.random.random((2, 2))*100)
        inputs = tf.constant(i, dtype=tf.float32)

        output = repeat(
            inputs,
            (2, 3))

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            o = s.run(output)

        c = np.repeat(i, 2, axis=0)
        c = np.repeat(c, 3, axis=1)

        np.testing.assert_array_equal(o, c)

    def test_nn_upsample(self):

        i = np.round(np.random.random((1, 1, 1, 1, 2))*100 - 50)
        inputs = tf.constant(i, dtype=tf.float32)

        output = upsample(
            inputs,
            [2, 3, 4],
            2,
            activation='relu',
            constant_upsample=True)[0]

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            o = s.run(output)

            assert_all_equal(o[0, 0, 0:2, 0:3, 0:4])
            assert_all_equal(o[0, 0, 0:2, 0:3, 4:8])
            assert_all_equal(o[0, 1, 0:2, 0:3, 0:4])
            assert_all_equal(o[0, 1, 0:2, 0:3, 4:8])
