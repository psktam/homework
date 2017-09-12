"""Implements behavioral cloning, which is just supervised learning algorithm"""
import argparse
import pickle

import tensorflow as tf
import gym
import numpy as np

import load_policy
import models


def generate_samples(policy_fn, environment, num_samples):
    """Generate samples in the given environment with the provided policy function"""
    if num_samples > environment.spec.timestep_limit:
        raise ValueError("Asking for {} steps. Can draw at most {} samples".format(num_samples,
                                                                                   environment.spec.timestep_limit))
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
        # environment.render()
        total_reward += reward

    return np.array(observations).squeeze(), np.array(actions).squeeze(), total_reward


def train(train_step, loss_function, name_mapping, session, expert_obs, expert_actions, batch_size,
          training_iters=1000000, rtol=1e-6):
    """
    Implements the training pipeline. The model is trained to try to match the given policy function, using the supplied
    openAI environment
    """
    expert_samples = len(expert_obs)
    last_cost = np.inf

    try:
        for training_iteration in range(training_iters):
            # Select random batches
            batch_samples = np.random.randint(0, expert_samples, batch_size)
            batch_obs, batch_actions = expert_obs[batch_samples], expert_actions[batch_samples]

            session.run(train_step, feed_dict={name_mapping['observations']: batch_obs,
                                               name_mapping['actions']: batch_actions})
            cost = session.run(loss_function, feed_dict={name_mapping['observations']: batch_obs,
                                                         name_mapping['actions']: batch_actions})
            relative_err = abs(cost - last_cost) / last_cost

            if training_iteration % 10 == 0:
                print("On training iteration {} out of {}".format(training_iteration, training_iters))
                print("    Current cost is {}".format(cost))
                print("    Relative error: {}".format(relative_err))

            last_cost = cost
            if relative_err <= rtol:
                break
    except KeyboardInterrupt:
        pass
    print("Finished training!")


def main():
    """Some main logic"""
    parser = argparse.ArgumentParser(description="Implement behavior cloning of the hopper expert policy")
    parser.add_argument("model", type=str)
    parser.add_argument("expert", type=str)
    parser.add_argument("--training_rate", "-r", type=float, default=0.01)
    parser.add_argument("--num_epochs", "-e", type=int, default=20)
    parser.add_argument("--relative_err", "-x", type=float, default=1e-6)
    parser.add_argument("--batch_size", "-b", type=int, default=100)
    args = parser.parse_args()

    expert_policy_file = 'experts/{}-v1.pkl'.format(args.expert)
    my_model = models.MODEL_REGISTRY[args.model](expert_policy_file)
    env = gym.make("{}-v1".format(args.expert))
    expert_policy = load_policy.load_policy(expert_policy_file)

    training_epochs = args.num_epochs
    loss_fun = my_model.loss_function
    train_step = tf.train.AdagradOptimizer(learning_rate=args.training_rate).minimize(loss_fun)

    all_obs, all_actions = [], []

    with tf.Session() as session:
        for epoch in range(training_epochs):
            print("Drawing samples, {} of {} epochs".format(epoch + 1, training_epochs))
            new_obs, new_actions, _ = generate_samples(expert_policy, env, 1000)
            all_obs.extend(new_obs)
            all_actions.extend(new_actions)
        all_obs = np.vstack(tuple(all_obs))
        all_actions = np.vstack(tuple(all_actions))

    with tf.Session() as session:
        print(tf.trainable_variables())
        tf.global_variables_initializer().run()
        train(train_step, loss_fun, my_model.varname_map, session, all_obs, all_actions, args.batch_size,
              rtol=args.relative_err)

        fname = input("Where do you want to save the derived policy? ")
        with open(fname, 'wb') as savefile:
            pickle.dump(my_model.save_variables(session), savefile)


if __name__ == '__main__':
    main()
