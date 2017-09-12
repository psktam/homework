"""Contains behavior-cloning models for the ant environment"""
import numpy as np
import tensorflow as tf

from .shared import register_model, ClonedModel


@register_model
class NaiveModel(ClonedModel):
    """I don't know what the network for the ant policy looks like, so I ain't going to try figureing it out"""
    name = 'naive_ant'

    def __init__(self, pickle_file):
        super(self.__class__, self).__init__(pickle_file)
        hidden_dim = 256

        self.varname_map = {}

        self.varname_map['observations'] = tf.placeholder(dtype=np.float64, shape=(None, self._obs_dim))
        self.varname_map['actions'] = tf.placeholder(dtype=np.float64, shape=(None, self._action_dim))

        hidden_layers = []
        hidden_biases = []
        last_output_dim = self._obs_dim

        for hidden_idx in range(3):
            hidden_layers.append(tf.Variable(initial_value=np.random.random((last_output_dim, hidden_dim))))
            hidden_biases.append(tf.Variable(initial_value=np.random.random((1, hidden_dim))))

            self.varname_map['hidden-{}'.format(hidden_idx)] = hidden_layers[-1]
            self.varname_map['bias-{}'.format(hidden_idx)] = hidden_biases[-1]

            last_output_dim = hidden_dim

        output_layer = tf.Variable(initial_value=np.random.random((hidden_dim, self._action_dim)))
        output_bias = tf.Variable(initial_value=np.random.random((1, self._action_dim)))
        self.varname_map['output'] = output_layer
        self.varname_map['output-bias'] = output_bias

    @property
    def loss_function(self):
        """Produce the network"""
        feedforward = (self.varname_map['observations'] - self._obs_mean) / (self._obs_std + 1e-6)
        hidden_layers = [self.varname_map['hidden-{}'.format(num)] for num in range(3)]
        biases = [self.varname_map['bias-{}'.format(num)] for num in range(3)]

        for weights, bias in zip(hidden_layers, biases):
            feedforward = tf.tanh(tf.matmul(feedforward, weights) + bias)
        feedforward = tf.matmul(feedforward, self.varname_map['output']) + self.varname_map['output-bias']

        cost = tf.losses.absolute_difference(labels=self.varname_map['actions'], predictions=feedforward)
        return cost

    def save_variables(self, session):
        """Save network in same format as encoded pickles"""
        pickle_dict = {'nonlin_type': 'tanh'}
        pickle_dict['GaussianPolicy'] = policy_dict = {}

        policy_dict['out'] = {'AffineLayer': {'W': session.run(self.varname_map['output']),
                                              'b': session.run(self.varname_map['output-bias'])}}
        policy_dict['hidden'] = {'FeedforwardNet': {
            'layer_0': {'AffineLayer': {
                'W': session.run(self.varname_map['hidden-0']),
                'b': session.run(self.varname_map['bias-0']),
            }},
            'layer_1': {'AffineLayer': {
                'W': session.run(self.varname_map['hidden-1']),
                'b': session.run(self.varname_map['bias-1'])
            }},
            'layer_2': {'AffineLayer': {
                'W': session.run(self.varname_map['hidden-2']),
                'b': session.run(self.varname_map['bias-2'])
            }}
        }}

        meansq = self._obs_mean ** 2.0 + self._obs_std ** 2.0
        policy_dict['obsnorm'] = {'Standardizer': {'mean_1_D': self._obs_mean, 'meansq_1_D': meansq}}
        policy_dict['logstdevs_1_Da'] = None

        return pickle_dict
