""" Helper functions for exercise 1.1 """
import numpy as np
import tensorflow as tf


def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    dim = tf.cast(tf.shape(x)[1], tf.float32)
    summation_terms = 2*log_std + ((x - mu)**2)/tf.exp(2*log_std)
    return -0.5*(tf.reduce_sum(summation_terms, axis=1)
                 + dim*np.log(2*np.pi))
