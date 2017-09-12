"""Contains behavior-cloning models of hopper"""
import numpy as np
import tensorflow as tf

from .shared import register_model, ClonedModel


@register_model
class CheatingModel(ClonedModel):
    """cheating solution, wherein I just recreate the exact structure of the expert policy"""
    name = 'cheating_hopper'

    def __init__(self, pickle_file):
        """Generate variables and map their names as well"""
        super(self.__class__, self).__init__(pickle_file)
        hidden_dim = 64

        observations = tf.placeholder(dtype=np.float64, shape=(None, self._obs_dim), name='observations')
        hidden_1 = tf.Variable(initial_value=np.random.random((self._obs_dim, hidden_dim)), name='hidden-1')
        bias_1 = tf.Variable(initial_value=np.random.random((1, hidden_dim)), name='bias-1')
        hidden_2 = tf.Variable(initial_value=np.random.random((hidden_dim, hidden_dim)), name='hidden-2')
        bias_2 = tf.Variable(initial_value=np.random.random((1, hidden_dim)), name='bias-2')
        output_layer = tf.Variable(initial_value=np.random.random((hidden_dim, self._action_dim)), name='output')
        output_bias = tf.Variable(initial_value=np.random.random((1, self._action_dim)), name='output-bias')
        actions = tf.placeholder(dtype=np.float64, shape=(None, self._action_dim), name='actions')

        self.varname_map = {'observations': observations, 'actions': actions,
                            'hidden-1': hidden_1, 'bias-1': bias_1,
                            'hidden-2': hidden_2, 'bias-2': bias_2,
                            'output': output_layer, 'output-bias': output_bias}

    @property
    def loss_function(self):
        """Produce the same structure"""
        feed_forward = (self.varname_map['observations'] - self._obs_mean) / self._obs_std
        hidden_layers = [self.varname_map[key] for key in ('hidden-1', 'hidden-2')]
        hidden_biases = [self.varname_map[key] for key in ('bias-1', 'bias-2')]
        for layer, bias in zip(hidden_layers, hidden_biases):
            feed_forward = tf.tanh(tf.matmul(feed_forward, layer) + bias)

        # Do not apply nonlinearity to the output layer
        feed_forward = tf.matmul(feed_forward, self.varname_map['output']) + self.varname_map['output-bias']

        # Hm, cost is interesting. Should this be a class-specific thing, or should that be something to decide when
        # optimizing? I'm going to go with the former for now.

        cost = tf.nn.l2_loss(self.varname_map['actions'] - feed_forward)
        return cost

    def save_variables(self, session):
        """Save the network in the same format as the encoded pickles"""
        pickle_dict = {'nonlin_type': 'tanh'}
        pickle_dict['GaussianPolicy'] = policy_dict = {}

        policy_dict['out'] = {'AffineLayer': {'W': session.run(self.varname_map['output']),
                                              'b': session.run(self.varname_map['output-bias'])}}
        policy_dict['hidden'] = {'FeedforwardNet': {
            'layer_0': {'AffineLayer': {
                'W': session.run(self.varname_map['hidden-1']),
                'b': session.run(self.varname_map['bias-1'])}},
            'layer_2': {'AffineLayer': {
                'W': session.run(self.varname_map['hidden-2']),
                'b': session.run(self.varname_map['bias-2'])}}}}
        meansq = self._obs_mean ** 2.0 + self._obs_std ** 2.0
        policy_dict['obsnorm'] = {'Standardizer': {'mean_1_D': self._obs_mean, 'meansq_1_D': meansq}}
        policy_dict['logstdevs_1_Da'] = None

        return pickle_dict
