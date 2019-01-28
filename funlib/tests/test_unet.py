from funlib.learn.tensorflow import models
import tensorflow as tf
import unittest


class TestUNet(unittest.TestCase):

    def test_creation(self):

        fmaps = tf.placeholder(tf.float32, shape=(1, 1, 100, 80, 48))

        unet, _, _ = models.unet(
            fmaps_in=fmaps,
            num_fmaps=3,
            fmap_inc_factors=2,
            downsample_factors=[[2, 2, 2], [2, 2, 2]],
            num_fmaps_out=5)

        assert unet.get_shape().as_list() == [1, 5, 60, 40, 8]
