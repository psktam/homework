"""Contains behavior-cloning models of hopper"""
import numpy as np
import tensorflow as tf

from .shared import register_model, ClonedModel


@register_model
class CheatingModel(ClonedModel):
    """cheating solution, wherein I just recreate the exact structure of the expert policy"""

    @property
    @classmethod
    def name(self):
        """gives name of this model"""
        return 'cheating_hopper'

    @property
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
