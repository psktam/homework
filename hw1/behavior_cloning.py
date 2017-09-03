"""Implements behavioral cloning, which is just supervised learning algorithm"""
import tensorflow as tf
import gym
import numpy as np

import load_policy
import tf_util
import models


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

    return np.array(observations).squeeze(), np.array(actions).squeeze(), total_reward


def train(loss_and_name_mapping, policy_function, environment,
          training_epochs=10, expert_samples=1000, training_iters=1000, rtol=1e-6):
    """
    Implements the training pipeline. The model is trained to try to match the given policy function, using the supplied
    openAI environment
    """
    loss_function, name_mapping = loss_and_name_mapping
    # Generate expert observations and actions
    expert_obs, expert_actions, _ = generate_samples(policy_function, environment, expert_samples)

    gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    last_cost = np.inf
    costs = []

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for training_iteration in range(training_iters):
            # Select random batches
            batch_samples = np.random.randint(0, expert_samples, 100)
            batch_obs, batch_actions = expert_obs[batch_samples], expert_actions[batch_samples]

            session.run(gradient_descent.minimize(loss=loss_function),
                        feed_dict={name_mapping['observations']: batch_obs, name_mapping['actions']: batch_actions})
            cost = session.run(loss_function, feed_dict={name_mapping['observations']: expert_obs,
                                                         name_mapping['actions']: expert_actions})
            costs.append(cost)
            relative_err = abs(cost - last_cost) / last_cost

            if training_iteration % 10 == 0:
                print("On training iteration {} out of {}".format(training_iteration, training_iters))
                print("    Current cost is {}".format(cost))
                print("    Relative error: {}".format(relative_err))

            last_cost = cost
            if relative_err <= rtol:
                break
    print("Finished training!")


def main():
    """Some main logic"""
    expert_policy_file = 'experts/Hopper-v1.pkl'
    my_model = models.MODEL_REGISTRY['cheating_hopper'](expert_policy_file)
    env = gym.make("Hopper-v1")
    expert_policy = load_policy.load_policy(expert_policy_file)
    train(my_model.loss_function, expert_policy, env)


if __name__ == '__main__':
    main()
