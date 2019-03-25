""" Helper functions for exercise 1.1 """
import numpy as np
import tensorflow as tf

from problem_set_1.exercise1_1_helper import gaussian_likelihood


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    """ Builds a multi-layer perceptron in Tensorflow.  """
    with tf.variable_scope('mlp'):
        for layer_no, hsz in enumerate(hidden_sizes[:-1]):
            with tf.variable_scope('layer_{}'.format(layer_no)):
                x = tf.layers.dense(x, units=hsz, activation=activation)
        with tf.variable_scope('output_layer'):
            x = tf.layers.dense(x, units=hidden_sizes[-1],
                                activation=output_activation)
    return x


def sample_actions(mu, log_std):
    """ Return a sample from a multivariate normal distribution parameterized
    by mean mu and log_std, the logs of the diagonals of the covariance matrix
    """
    std = tf.exp(log_std)
    noise = std*tf.random.normal(shape=tf.shape(mu))
    return mu + noise


def make_log_std_var(dim):
    """ Initialize diagonal entries of covariance matrix """
    init = tf.constant(-0.5, shape=[1, dim], dtype=tf.float32)
    return tf.get_variable('log_std', initializer=init, trainable=True)


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation,
                        action_space):
    """ Builds symbols to sample actions and compute log-probs of actions.  """
    act_dim = np.prod(action_space.shape)
    mlp_hidden_sizes = hidden_sizes + (act_dim,)
    mu = mlp(x, hidden_sizes=mlp_hidden_sizes, activation=activation,
             output_activation=output_activation)
    log_std = make_log_std_var(act_dim)
    pi = sample_actions(mu, log_std)
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi
