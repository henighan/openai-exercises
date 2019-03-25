""" Test for exercise 1.2 """
from unittest.mock import Mock
import tensorflow as tf
import numpy as np

from problem_set_1 import exercise1_2_helper

E12_PATH = 'problem_set_1.exercise1_2_helper.'


class TestEx12(tf.test.TestCase):

    def test_mlp_smoke(self):
        """ Smoke test mlp """
        batch_size = 3
        input_dim = 2
        output_dim = 4
        x = np.zeros(shape=[batch_size, input_dim], dtype=np.float32)
        with self.cached_session() as sess:
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
            ret = exercise1_2_helper.mlp(x_ph, hidden_sizes=(output_dim,))
            sess.run(tf.global_variables_initializer())
            n_trainable_variables = 2 # 1 kernel and 1 bias
            trainable_variables = tf.trainable_variables()
            self.assertEqual(len(trainable_variables), n_trainable_variables)
            ret_eval = sess.run(ret, feed_dict={x_ph: x})
            self.assertEqual(ret_eval.shape, (batch_size, output_dim))

    def test_mlp_multiple_layers(self):
        """ test mlp makes multiple layers, with weights of the correct
        shapes """
        batch_size = 3
        input_dim = 2
        hidden_sizes = (5, 4, 3)
        x = np.zeros(shape=[batch_size, input_dim], dtype=np.float32)
        with self.cached_session() as sess:
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
            ret = exercise1_2_helper.mlp(x_ph, hidden_sizes=hidden_sizes)
            sess.run(tf.global_variables_initializer())
            n_trainable_variables = 6 # 3 kernels and 3 bias
            ret_eval = sess.run(ret, feed_dict={x_ph: x})
            self.assertEqual(ret_eval.shape, (batch_size, 3))
            trainable_variables = sess.run(tf.trainable_variables())
            variable_shapes = [var.shape for var in trainable_variables]
            self.assertEqual(len(trainable_variables), n_trainable_variables)
            # kernels
            self.assertIn((2, 5), variable_shapes)
            self.assertIn((5, 4), variable_shapes)
            self.assertIn((4, 3), variable_shapes)
            # biases
            self.assertIn((5,), variable_shapes)
            self.assertIn((4,), variable_shapes)
            self.assertIn((3,), variable_shapes)


    def test_make_log_std_smoke(self):
        """ smoke test make_log_std """
        dim = 12
        with self.cached_session() as sess:
            ret = exercise1_2_helper.make_log_std_var(dim)
            n_trainable_variables = len(tf.trainable_variables())
            # ensure log_std is trainable
            self.assertEqual(n_trainable_variables, 1)
            sess.run(tf.global_variables_initializer())
            np.testing.assert_array_equal(ret.eval(), -0.5*np.ones([1, dim]))


    def test_sample_actions_smoke(self):
        """ smoke test sample_actions """
        batch_size = 5
        dim = 3
        mu = np.zeros(shape=[5, dim])
        log_std = np.zeros(shape=[1, dim])
        with self.cached_session() as sess:
            mu_ph = tf.placeholder(dtype=tf.float32, shape=[None, dim])
            log_std_ph = tf.placeholder(dtype=tf.float32, shape=[1, dim])
            ret = exercise1_2_helper.sample_actions(mu_ph, log_std_ph)
            ret_eval = sess.run(ret, feed_dict={mu_ph: mu, log_std_ph: log_std})
            self.assertEqual(ret_eval.shape, (batch_size, dim))

    def test_mlp_gaussian_policy_smoke(self):
        """ smoke test of mlp_gaussian_policy """
        batch_size = 5
        obs_dim = 3
        act_dim = 2
        action_space = Mock()
        action_space.contains.return_value = True
        action_space.shape = (act_dim,)
        hidden_sizes = (4,)
        x = np.zeros(shape=[batch_size, obs_dim])
        a = np.zeros(shape=[batch_size, act_dim])
        with self.cached_session() as sess:
            x_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim])
            a_ph = tf.placeholder(dtype=tf.float32, shape=[None, act_dim])
            ret = exercise1_2_helper.mlp_gaussian_policy(
                x_ph, a_ph, hidden_sizes=hidden_sizes, activation=tf.tanh,
                output_activation=tf.tanh, action_space=action_space)
            sess.run(tf.global_variables_initializer())
            ret_pi, ret_logp, ret_logp_pi = sess.run(
                ret, feed_dict={x_ph: x, a_ph: a})
            self.assertEqual(ret_pi.shape, (batch_size, act_dim))
            self.assertEqual(ret_logp.shape, (batch_size,))
            self.assertEqual(ret_logp_pi.shape, (batch_size,))
