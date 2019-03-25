""" Helper functions for exercise 1.3 """
import numpy as np
import tensorflow as tf


def get_clipped_noise(shape, noise_scale=0.1, noise_clip=0.5):
    """ get normally distributed noise with std=noise_scale, clipped to +-
    noise_clip """
    noise = noise_scale*tf.random.normal(shape=shape)
    return tf.clip_by_value(noise, -noise_clip, noise_clip)


def get_pi_noise_clipped(pi, noise_scale=0.1, noise_clip=0.5, act_limit=10.):
    """ Add clipped noise to sampled target action, and then clip to stay
    within the valid value of the action """
    clipped_noise = get_clipped_noise(
        tf.shape(pi), noise_scale=noise_scale, noise_clip=noise_clip)
    pi_noise = pi + clipped_noise
    pi_noise_clipped = tf.clip_by_value(pi_noise, -act_limit, act_limit)
    return pi_noise_clipped


def get_q_target(q1, q2, r, d, gamma=0.99):
    """ calculate q target """
    return r + gamma*(1 - d)*tf.minimum(q1, q2, name='min_q')
