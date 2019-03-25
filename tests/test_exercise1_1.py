""" Tests for exercise 1.1 """
import numpy as np
import tensorflow as tf

from problem_set_1 import exercise1_1_helper


class TestEx11(tf.test.TestCase):

    def test_gaussian_likelihood_log_std_2d(self):
        """ test for guassian likelihood when log_std is 2 dimensional """
        batch_size = 3
        dim = 2
        x = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        mu = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        log_std = tf.constant(1, dtype=tf.float32, shape=[batch_size, dim])
        ret = exercise1_1_helper.gaussian_likelihood(x, mu, log_std)
        expected = (- dim - 0.5*dim*np.log(2*np.pi))*np.ones([batch_size])
        with self.cached_session():
            self.assertAllClose(expected, ret.eval())

    def test_gaussian_likelihood_log_std_1d(self):
        """ test for guassian likelihood when log_std is 1 dimensional """
        batch_size = 3
        dim = 2
        x = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        mu = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        log_std = tf.constant(1, dtype=tf.float32, shape=[dim])
        ret = exercise1_1_helper.gaussian_likelihood(x, mu, log_std)
        expected = (- dim - 0.5*dim*np.log(2*np.pi))*np.ones([batch_size])
        with self.cached_session():
            self.assertAllClose(expected, ret.eval())

    def test_gaussian_likelihood_non_zero_difference(self):
        """ test for guassian likelihood when log_std is 1 dimensional """
        batch_size = 3
        dim = 2
        x = tf.constant(2, dtype=tf.float32, shape=[batch_size, dim])
        mu = tf.constant(0, dtype=tf.float32, shape=[batch_size, dim])
        log_std = tf.constant(1, dtype=tf.float32, shape=[dim])
        ret = exercise1_1_helper.gaussian_likelihood(x, mu, log_std)
        expected_summation_term = -0.5*dim*(4/np.e**2 + 2)
        expected_scalar_term = -0.5*dim*np.log(2*np.pi)
        expected = (expected_summation_term
                    + expected_scalar_term)*np.ones([batch_size])
        with self.cached_session():
            self.assertAllClose(expected, ret.eval())
