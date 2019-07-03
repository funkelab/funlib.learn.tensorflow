from funlib.learn.tensorflow import models
import tensorflow as tf
import pytest
import warnings
warnings.filterwarnings("error")


class TestUNet(tf.test.TestCase):

    def test_creation(self):
        with self.test_session():
            fmaps = tf.placeholder(tf.float32, shape=(1, 1, 100, 80, 48))

            unet, _, _ = models.unet(
                fmaps_in=fmaps,
                num_fmaps=3,
                fmap_inc_factors=2,
                downsample_factors=[[2, 2, 2], [2, 2, 2]],
                num_fmaps_out=5)

            assert unet.get_shape().as_list() == [1, 5, 60, 40, 8]

    def test_shape_warning(self):
        with self.test_session():
            # Should raise warning
            fmaps = tf.placeholder(tf.float32, shape=(1, 1, 100, 80, 48))

            with pytest.raises(Exception):
                unet, _, _ = models.unet(
                    fmaps_in=fmaps,
                    num_fmaps=3,
                    fmap_inc_factors=2,
                    downsample_factors=[[2, 3, 2], [2, 2, 2]],
                    num_fmaps_out=5)

    def test_num_heads_option(self):
        with self.test_session():
            fmaps = tf.placeholder(tf.float32, shape=(1, 1, 100, 80, 48))
            unet, _, _ = models.unet(
                fmaps_in=fmaps,
                num_fmaps=3,
                fmap_inc_factors=2,
                downsample_factors=[[2, 2, 2], [2, 2, 2]],
                num_fmaps_out=5,
                num_heads=3)

            assert unet[0].get_shape().as_list() == [1, 5, 60, 40, 8]
            assert unet[1].get_shape().as_list() == [1, 5, 60, 40, 8]
            assert unet[2].get_shape().as_list() == [1, 5, 60, 40, 8]
            assert len(unet) == 3

    def test_crop_to_factor(self):

        with self.test_session():

            with tf.variable_scope('fail'):
                with pytest.raises(AssertionError):
                    fmaps = tf.placeholder(
                        tf.float32,
                        shape=(1, 1, 22, 25, 25))
                    unet, _, _ = models.unet(
                        fmaps_in=fmaps,
                        num_fmaps=1,
                        fmap_inc_factors=1,
                        downsample_factors=[[3, 3, 3]])

            with tf.variable_scope('minimal'):
                fmaps = tf.placeholder(tf.float32, shape=(1, 1, 25, 25, 25))
                unet, _, _ = models.unet(
                    fmaps_in=fmaps,
                    num_fmaps=1,
                    fmap_inc_factors=2,
                    downsample_factors=[[3, 3, 3]])

                assert unet.get_shape().as_list() == [1, 1, 3, 3, 3]

            with tf.variable_scope('nested'):
                i = 102
                fmaps = tf.placeholder(tf.float32, shape=(1, 1, i, i, i))
                unet, _, _ = models.unet(
                    fmaps_in=fmaps,
                    num_fmaps=1,
                    fmap_inc_factors=1,
                    kernel_size_up=[[3], [3, 3]],
                    downsample_factors=[[2, 2, 2], [3, 3, 3]])

                assert unet.get_shape().as_list() == [1, 1, 48, 48, 48]
