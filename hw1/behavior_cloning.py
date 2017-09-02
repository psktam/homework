"""Implements behavioral cloning, which is just supervised learning algorithm"""
import tensorflow as tf

import tf_util


def generate_samples(policy_fn, environment, num_samples):
    """Generate samples in the given environment with the provided policy function"""
    if num_samples > environment.spec.timestep_limit:
        raise ValueError("Asking for {} steps. Can draw at most {} samples".format(num_samples,
                                                                                   environment.spec.timestep_limit))

    with tf.Session():
        tf_util.initialize()
        total_reward = 0.0
        observations = []
        actions = []

        current_observation = environment.reset()

        for _ in range(num_samples):
            action = policy_fn(current_observation[None, :])
            observations.append(current_observation)
            actions.append(action)

            # Step into the next time
            current_observation, reward, _, _ = environment.step(action)
            total_reward += reward

    return observations, actions, total_reward
