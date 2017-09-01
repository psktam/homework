"""Some simple sanity checks on my functions"""
import gym
import nose.tools as nt

import behavior_cloning as bc
import load_policy


def test_draw_samples():
    """Just make sure it doesn't raise exceptions"""
    n_samples = 1000
    policy_fun = load_policy.load_policy('experts/Hopper-v1.pkl')
    environment = gym.make("Hopper-v1")
    samples = bc.generate_samples(policy_fun, environment, n_samples)

    nt.assert_equals(len(samples[0]), n_samples)
    nt.assert_equals(len(samples[1]), n_samples)
    nt.assert_greater(samples[2], 0)
