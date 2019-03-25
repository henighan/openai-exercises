"""

Exercise 1.2: PPO Gaussian Policy

Implement an MLP diagonal Gaussian policy for PPO. 

Log-likelihoods will be computed using your answer to Exercise 1.1,
so make sure to complete that exercise before beginning this one.

"""
import tensorflow as tf
import numpy as np

from problem_set_1.exercise1_2_helper import mlp_gaussian_policy

EPS = 1e-8


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """

    from spinup import ppo
    from spinup.exercises.common import print_result
    import gym
    import os
    import pandas as pd
    import psutil
    import time

    logdir = "/tmp/experiments/%i"%int(time.time())
    ppo(env_fn = lambda : gym.make('InvertedPendulum-v2'),
        ac_kwargs=dict(policy=mlp_gaussian_policy, hidden_sizes=(64,)),
        steps_per_epoch=4000, epochs=20, logger_kwargs=dict(output_dir=logdir))

    # Get scores from last five epochs to evaluate success.
    data = pd.read_table(os.path.join(logdir,'progress.txt'))
    last_scores = data['AverageEpRet'][-5:]

    # Your implementation is probably correct if the agent has a score >500,
    # or if it reaches the top possible score of 1000, in the last five epochs.
    correct = np.mean(last_scores) > 500 or np.max(last_scores)==1e3
    print_result(correct)
