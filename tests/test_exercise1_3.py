""" Test for exercise 1.3 """
from unittest.mock import patch
import tensorflow as tf
import numpy as np

from problem_set_1 import exercise1_3_helper

E13_PATH = 'problem_set_1.exercise1_3_helper.'


class TestEx13(tf.test.TestCase):

    def test_get_clipped_noise_smoke(self):
        """ smoke test get_clipped_noise """
        batch_size = 4
        act_dim = 2
        pi = tf.constant(0, shape=[batch_size, act_dim])
        with self.cached_session():
            ret = exercise1_3_helper.get_clipped_noise(tf.shape(pi))
            self.assertEqual(ret.eval().shape, (batch_size, act_dim))

    def test_get_clipped_noise_clips(self):
        """ Test that the noise gets clipped by setting noise >> clip """
        batch_size = 4
        act_dim = 2
        eps = 1e-4
        pi = tf.constant(0, shape=[batch_size, act_dim])
        with self.cached_session():
            ret = exercise1_3_helper.get_clipped_noise(
                tf.shape(pi), noise_scale=10., noise_clip=0.1)
            ret_eval = ret.eval()
            min_ret = np.min(ret_eval)
            max_ret = np.max(ret_eval)
            self.assertGreaterEqual(min_ret, -0.1 - eps)
            self.assertLessEqual(max_ret, 0.1 + eps)
            self.assertEqual(ret.eval().shape, (batch_size, act_dim))


    @patch(E13_PATH + 'get_clipped_noise')
    def test_get_pi_noise_clipped_smoke(self, get_noise_mock):
        """ smoke test for get pi noise clipped """
        batch_size = 4
        act_dim = 1
        pi = np.array([[2], [1], [-1], [-2]], dtype=np.float32)
        expected = np.array([[1.5], [1], [-1], [-1.5]], dtype=np.float32)
        get_noise_mock.return_value = tf.constant(
            0, shape=[batch_size, act_dim], dtype=tf.float32)
        with self.cached_session() as sess:
            pi_ph = tf.placeholder(shape=[None, act_dim], dtype=tf.float32)
            ret = exercise1_3_helper.get_pi_noise_clipped(
                pi_ph, act_limit=1.5)
            ret_eval = sess.run(ret, feed_dict={pi_ph: pi})
            np.testing.assert_array_equal(ret_eval, expected)


    def test_get_q_target_smoke(self):
        """ smoke test get_q_target """
        q1 = np.array([0.1, 0.5, 0.2, 0.4])
        q2 = np.array([0.3, 0.2, 0.3, 0.1])
        min_q = np.array([0.1, 0.2, 0.2, 0.1])
        r = np.ones_like(q1)
        d = np.array([False, False, False, True]).astype(np.float32)
        gamma = 0.9
        expected = 1 + gamma*min_q
        expected[-1] = 1
        with self.cached_session() as sess:
            q1_ph = tf.placeholder(dtype=tf.float32, shape=[None])
            q2_ph = tf.placeholder(dtype=tf.float32, shape=[None])
            r_ph = tf.placeholder(dtype=tf.float32, shape=[None])
            d_ph = tf.placeholder(dtype=tf.float32, shape=[None])
            ret = exercise1_3_helper.get_q_target(
                q1_ph, q2_ph, r_ph, d_ph, gamma)
            ret_eval = sess.run(ret, feed_dict={
                q1_ph: q1, q2_ph: q2, r_ph: r, d_ph: d})
            self.assertEqual(ret_eval.shape, (4,))
            np.testing.assert_almost_equal(ret_eval, expected)
