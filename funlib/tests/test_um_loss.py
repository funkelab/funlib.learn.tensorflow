from funlib.learn.tensorflow.losses import ultrametric_loss_op
import numpy as np
import tensorflow as tf
import unittest


class TestUmLoss(unittest.TestCase):

    def test_zero(self):

        embedding = np.zeros((3, 10, 10, 10), dtype=np.float32)
        segmentation = np.ones((10, 10, 10), dtype=np.int64)

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            add_coordinates=False,
            name='um_test_zero')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 0)
            self.assertEqual(np.sum(distances), 0)

    def test_simple(self):

        embedding = np.array(
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]],
            dtype=np.float32).reshape((1, 1, 3, 3))

        segmentation = np.array(
            [[1, 1, 1],
             [2, 2, 2],
             [3, 3, 3]],
            dtype=np.int64).reshape((1, 3, 3))

        # number of positive pairs: 3*3 = 9
        # number of negative pairs: 3*3*3 = 27
        # total number of pairs: 9*8/2 = 36

        # loss on positive pairs: 9*1 = 9
        # loss on negative pairs: 27*1 = 27
        # total loss = 36
        # total loss per edge = 1

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            add_coordinates=False,
            balance=False,
            name='um_test_simple_unbalanced')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 1.0)
            self.assertAlmostEqual(np.sum(distances), 8, places=4)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            add_coordinates=False,
            name='um_test_simple_balanced')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 2.0)
            self.assertAlmostEqual(np.sum(distances), 8, places=4)

    def test_quadrupel_loss(self):

        embedding = np.array(
            [[0, 1, 2],
             [4, 5, 6],
             [8, 9, 10]],
            dtype=np.float32).reshape((1, 1, 3, 3))

        segmentation = np.array(
            [[1, 1, 1],
             [2, 2, 2],
             [3, 3, 3]],
            dtype=np.int64).reshape((1, 3, 3))

        # number of positive pairs: 3*3 = 9
        # number of negative pairs: 3*3*3 = 27
        # number of quadrupels: 9*27 = 243

        # loss per quadrupel: max(0, d(p) - d(n) + alpha)^2 = (1 - 2 + 3)^2 = 4

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=3,
            add_coordinates=False,
            quadrupel_loss=True,
            name='um_test_quadrupel_loss')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            print(loss)
            print(emst)
            print(edges_u)
            print(edges_v)
            print(distances)

            self.assertEqual(loss, 4.0)
            self.assertAlmostEqual(np.sum(distances), 10, places=4)
