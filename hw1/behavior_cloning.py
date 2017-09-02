"""Implements behavioral cloning, which is just supervised learning algorithm"""
import abc
import pickle

import numpy as np
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


# This next section will implement the code that will attempt to clone the behavior of the hopper expert policy. The
# provided policy uses two hidden layers, each with 64 nodes, and the output space is 3-dimensional. Activations emitted
# by each hidden layer is normalized via a tanh function. The input space is 11-dimensional
HOPPER_MODEL_REGISTRY = {}


def register_model(modelclass):
    """registers models in the hopper model registry"""
    if modelclass.name in HOPPER_MODEL_REGISTRY:
        raise ValueError("Cannot add model with name {}, as it already has been registered".format(modelclass.name))
    HOPPER_MODEL_REGISTRY[modelclass.name] = modelclass


class ClonedModel(object):
    """Simple class to clone the behavior of the Hopper-v1 model"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, pickle_file):
        """Load the pickle and store the observation mean and standard deviations"""
        with open(pickle_file, 'rb') as pf:
            policy = pickle.load(pf)

        self._obs_mean = policy['GaussianPolicy']['obsnorm']['Standardizer']['mean_1_D']
        self._obs_dim = self._obs_mean.shape[1]
        obs_meansq = policy['GaussianPolicy']['obsnorm']['Standardizer']['meansq_1_D']
        self._obs_std = np.sqrt(obs_meansq - self._obs_mean ** 2.0) + 1e-6
        self._action_dim = policy['GaussianPolicy']['out']['AffineLayer']['b'].shape[1]

    def normalize(self, observations):
        """normalizes the observations over prior knowledge of the mean and standard deviation"""
        return (observations - self._obs_mean) / self._obs_std

    @property
    @abc.abstractclassmethod
    def name(self):
        """A unique identifier for the model, to be added to the model registry"""

    @property
    @abc.abstractmethod
    def loss_function(self):
        """Should return tensor that can be worked on with tensorflow's optimization classes"""


@register_model
class CheatingModel(ClonedModel):
    """cheating solution, wherein I just recreate the exact structure of the expert policy"""

    @property
    @classmethod
    def name(self):
        """gives name of this model"""
        return 'cheating'

    @property
    @abc.abstractmethod
    def loss_function(self):
        """Produce the same structure"""
        hidden_dim = 64

        observations = tf.placeholder(dtype=np.float64, shape=(None, self._obs_dim), name='observations')
        hidden_1 = tf.Variable(initial_value=np.random.random((self._obs_dim, hidden_dim)))
        bias_1 = tf.Variable(initial_value=np.random.random((1, hidden_dim)))
        hidden_2 = tf.Variable(initial_value=np.random.random((hidden_dim, hidden_dim)))
        bias_2 = tf.Variable(initial_value=np.random.random((1, hidden_dim)))
        output_layer = tf.Variable(initial_value=np.random.random((hidden_dim, self._action_dim)))
        output_bias = tf.Variable(initial_value=np.random.random((1, self._action_dim)))

        feed_forward = observations
        for layer, bias in zip((hidden_1, hidden_2), (bias_1, bias_2)):
            feed_forward = tf.tanh(tf.matmul(feed_forward, layer) + bias)

        # Do not apply nonlinearity to the output layer
        feed_forward = tf.matmul(feed_forward, output_layer) + output_bias

        # Hm, cost is interesting. Should this be a class-specific thing, or should that be something to decide when
        # optimizing? I'm going to go with the former for now.
        actions = tf.placeholder(dtype=np.float64, shape=(None, self._action_dim), name='actions')
        cost = tf.nn.l2_loss(actions - feed_forward)
        return cost
