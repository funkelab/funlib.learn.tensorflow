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

    def test_background(self):

        embedding = np.array(
            [[0, 1, 2],
             [4, 5, 6],
             [8, 9, 10]],
            dtype=np.float32).reshape((1, 1, 3, 3))

        segmentation = np.array(
            [[1, 1, 1],
             [0, 0, 0],
             [3, 3, 3]],
            dtype=np.int64).reshape((1, 3, 3))

        # number of positive pairs: 2*3 = 6
        # number of negative pairs: 3*3*3 = 27
        # number of background pairs: 3
        # total number of pairs (without background pairs): 33

        # loss on positive pairs: 6*1 = 6
        # loss on negative pairs: 27*2^2 = 108
        # total loss = 114
        # total loss per pair = 3.455
        # total loss per pos pair = 1
        # total loss per neg pair = 4

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=4,
            add_coordinates=False,
            balance=False,
            name='um_test_background')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertAlmostEqual(loss, 3.4545, places=4)
            self.assertAlmostEqual(np.sum(distances), 10, places=4)

    def test_mask(self):

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

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        # empty mask

        mask = np.zeros((1, 3, 3), dtype=np.bool)
        mask = tf.constant(mask)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            mask=mask,
            alpha=2,
            add_coordinates=False,
            name='um_test_simple_unbalanced')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 0.0)
            self.assertAlmostEqual(np.sum(distances), 0, places=4)

        # mask with only one point

        mask = np.zeros((1, 3, 3), dtype=np.bool)
        mask[0, 1, 1] = True
        mask = tf.constant(mask)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            mask=mask,
            alpha=2,
            add_coordinates=False,
            name='um_test_simple_unbalanced')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 0.0)
            self.assertAlmostEqual(np.sum(distances), 0, places=4)

        # mask with two points

        mask = np.zeros((1, 3, 3), dtype=np.bool)
        mask[0, 1, 1] = True
        mask[0, 0, 0] = True
        mask = tf.constant(mask)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            mask=mask,
            alpha=5,
            add_coordinates=False,
            name='um_test_simple_unbalanced')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 1.0)
            self.assertAlmostEqual(np.sum(distances), 4.0, places=4)

    def test_constrained(self):

        embedding = np.array(
            [[0, 1, 101],
             [2, 3, 4],
             [5, 6, 7]],
            dtype=np.float32).reshape((1, 1, 3, 3))

        segmentation = np.array(
            [[1, 1, 1],
             [2, 2, 2],
             [3, 3, 3]],
            dtype=np.int64).reshape((1, 3, 3))

        # number of positive pairs: 3*3 = 9
        # number of negative pairs: 3*3*3 = 27
        # total number of pairs: 9*8/2 = 36

        # loss on positive pairs: 6*1 + 1 + 2*100^2 = 20007
        # loss on negative pairs: 27*1 = 27
        # total loss: 1/9*20007 + 1/27*27 = 2224

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            add_coordinates=False,
            constrained_emst=True,
            name='um_test_constrained')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertAlmostEqual(loss, 2224.0, places=1)
            self.assertAlmostEqual(np.sum(distances), 107, places=4)

    def test_ambiguous_unkown(self):

        embedding = np.array(
            [[0, 1]],
            dtype=np.float32).reshape((1, 1, 1, 2))

        segmentation = np.array(
            [[-1, 0]],
            dtype=np.int64).reshape((1, 1, 2))

        # number of positive pairs: 0
        # number of negative pairs: 1
        # total number of pairs: 0

        # loss on positive pairs: 0
        # loss on negative pairs: 1
        # total loss: 1

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            balance=True,
            add_coordinates=False,
            constrained_emst=True,
            name='um_test_constrained')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertAlmostEqual(loss, 1, places=3)
            self.assertAlmostEqual(np.sum(distances), 1, places=4)

        embedding = np.array(
            [[0, 1]],
            dtype=np.float32).reshape((1, 1, 1, 2))

        segmentation = np.array(
            [[0, -1]],
            dtype=np.int64).reshape((1, 1, 2))

        # number of positive pairs: 0
        # number of negative pairs: 1
        # total number of pairs: 0

        # loss on positive pairs: 0
        # loss on negative pairs: 1
        # total loss: 1

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            balance=True,
            add_coordinates=False,
            constrained_emst=True,
            name='um_test_constrained')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertAlmostEqual(loss, 1, places=3)
            self.assertAlmostEqual(np.sum(distances), 1, places=4)

    def test_ambiguous_known(self):
        embedding = np.array(
            [[0, 1]],
            dtype=np.float32).reshape((1, 1, 1, 2))

        segmentation = np.array(
            [[-1, 1]],
            dtype=np.int64).reshape((1, 1, 2))

        # number of positive pairs: 0
        # number of negative pairs: 0
        # total number of pairs: 0

        # loss on positive pairs: 0
        # loss on negative pairs: 0
        # total loss: 0

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            balance=True,
            add_coordinates=False,
            constrained_emst=True,
            name='um_test_constrained')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertAlmostEqual(loss, 0, places=3)
            self.assertAlmostEqual(np.sum(distances), 1, places=4)

        embedding = np.array(
            [[0, 1]],
            dtype=np.float32).reshape((1, 1, 1, 2))

        segmentation = np.array(
            [[1, -1]],
            dtype=np.int64).reshape((1, 1, 2))

        # number of positive pairs: 0
        # number of negative pairs: 0
        # total number of pairs: 0

        # loss on positive pairs: 0
        # loss on negative pairs: 0
        # total loss: 0

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            balance=True,
            add_coordinates=False,
            constrained_emst=True,
            name='um_test_constrained')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertAlmostEqual(loss, 0, places=3)
            self.assertAlmostEqual(np.sum(distances), 1, places=4)

    def test_ambiguous_ambiguous(self):
        embedding = np.array(
            [[0, 1]],
            dtype=np.float32).reshape((1, 1, 1, 2))

        segmentation = np.array(
            [[-1, -1]],
            dtype=np.int64).reshape((1, 1, 2))

        # number of positive pairs: 0
        # number of negative pairs: 0
        # total number of pairs: 0

        # loss on positive pairs: 0
        # loss on negative pairs: 0
        # total loss: 0

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            alpha=2,
            balance=True,
            add_coordinates=False,
            constrained_emst=True,
            name='um_test_constrained')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertAlmostEqual(loss, 0, places=3)
            self.assertAlmostEqual(np.sum(distances), 1, places=4)

    def test_large_example(self):
        """
        nodes:
        [[a, b, c],
         [d, e, f],
         [g, h, i],
         [j, k, l]]
        mst edges (unconstrained), dist, ratio_pos, ratio_neg
        de,                        0     0          0
        ef,                        0     1          0
        jk,                        0     0          0
        ab,                        0.5   0          1
        bc,                        0.5   1          1
        gh,                        1     0          0
        hi,                        1     1          0
        kl,                        1     2          0
        dj,                        1.5   0          9
        fj,                        2     0          13
        gl,                        2     0          24
        mst edges (constrained),   dist, ratio_pos, ratio_neg
        de,                        0     1          0
        jk,                        0     1          0
        ac,                        1     1          0
        gh,                        1     2          0
        hi,                        1     0          0
        kl,                        1     0          0
        bk,                        4     0          0
        ef,                        0     0          0
        ab,                        0.5   0          8
        dj,                        1.5   0          16
        gl,                        2     0          24
        """

        embedding = np.array(
            [[0.5, 1, 1.5],
             [3.5, 3.5, 3.5],
             [8, 9, 10],
             [5, 5, 6]],
            dtype=np.float32).reshape((1, 1, 4, 3))
        embedding = tf.constant(embedding, dtype=tf.float32)

        segmentation = np.array(
            [[1, 0, 1],
             [2, 2, -1],
             [3, 3, 3],
             [0, 0, 0]],
            dtype=np.int64).reshape((1, 4, 3))
        segmentation = tf.constant(segmentation)

        loss_unbalanced_unconstrained = ultrametric_loss_op(
            embedding,
            segmentation,
            constrained_emst=False,
            balance=False,
            alpha=2,
            add_coordinates=False,
            name='um_test_unbalanced_unconstrained')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(
                loss_unbalanced_unconstrained
            )

            self.assertAlmostEqual(loss, 10/53, places=4)
            self.assertAlmostEqual(np.sum(distances), 9.5, places=4)

        loss_unbalanced_constrained = ultrametric_loss_op(
            embedding,
            segmentation,
            constrained_emst=True,
            balance=False,
            alpha=2,
            add_coordinates=False,
            name="um_test_unbalanced_constrained"
        )

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(
                loss_unbalanced_constrained
            )

            self.assertAlmostEqual(loss, 26/53, places=4)
            self.assertAlmostEqual(np.sum(distances), 12, places=4)

        loss_balanced_unconstrained = ultrametric_loss_op(
            embedding,
            segmentation,
            constrained_emst=False,
            balance=True,
            alpha=2,
            add_coordinates=False,
            name="um_test_balanced_unconstrained"
        )

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(
                loss_balanced_unconstrained
            )

            self.assertAlmostEqual(loss, 0.790625, places=4)
            self.assertAlmostEqual(np.sum(distances), 9.5, places=4)

        loss_balanced_constrained = ultrametric_loss_op(
            embedding,
            segmentation,
            constrained_emst=True,
            balance=True,
            alpha=2,
            add_coordinates=False,
            name="um_test_balanced_constrained"
        )

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(
                loss_balanced_constrained
            )

            self.assertAlmostEqual(loss, 1.258333, places=4)
            self.assertAlmostEqual(np.sum(distances), 12, places=4)

    def test_constrained_mask(self):

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

        embedding = tf.constant(embedding, dtype=tf.float32)
        segmentation = tf.constant(segmentation)

        # empty mask

        mask = np.zeros((1, 3, 3), dtype=np.bool)
        mask = tf.constant(mask)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            mask=mask,
            constrained_emst=True,
            alpha=2,
            add_coordinates=False,
            name='um_test_constrained_mask')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 0.0)
            self.assertAlmostEqual(np.sum(distances), 0, places=4)

        # mask with only one point

        mask = np.zeros((1, 3, 3), dtype=np.bool)
        mask[0, 1, 1] = True
        mask = tf.constant(mask)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            mask=mask,
            constrained_emst=True,
            alpha=2,
            add_coordinates=False,
            name='um_test_constrained_mask')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 0.0)
            self.assertAlmostEqual(np.sum(distances), 0, places=4)

        # mask with two points

        mask = np.zeros((1, 3, 3), dtype=np.bool)
        mask[0, 1, 1] = True
        mask[0, 0, 0] = True
        mask = tf.constant(mask)

        loss = ultrametric_loss_op(
            embedding,
            segmentation,
            mask=mask,
            constrained_emst=True,
            alpha=5,
            add_coordinates=False,
            name='um_test_constrained_mask')

        with tf.Session() as s:

            s.run(tf.global_variables_initializer())
            loss, emst, edges_u, edges_v, distances = s.run(loss)

            self.assertEqual(loss, 1.0)
            self.assertAlmostEqual(np.sum(distances), 4.0, places=4)

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

            self.assertEqual(loss, 4.0)
            self.assertAlmostEqual(np.sum(distances), 10, places=4)
