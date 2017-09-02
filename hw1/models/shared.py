"""contains base code for models"""
import abc
import pickle
import numpy as np


MODEL_REGISTRY = {}


def register_model(modelclass):
    """registers models in the hopper model registry"""
    if modelclass.name in MODEL_REGISTRY:
        raise ValueError("Cannot add model with name {}, as it already has been registered".format(modelclass.name))
    MODEL_REGISTRY[modelclass.name] = modelclass


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
