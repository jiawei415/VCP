from __future__ import with_statement
import tensorflow as tf
from vcp.algo.util import store_args, nn

class ActorCritic: 
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, last_hidden, layers, k_heads, share_network, **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (vcp.algo.Normalizer): normalizer for observations
            g_stats (vcp.algo.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u'] 

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor
        input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])  # for critic
        self._input_Q = input_Q  # exposed for tests

        if self.share_network:
            self.share_layers = self.layers - 1
            self.layers = 1
        else:
            self.share_layers = 0
        self.pi_tf_dict, self.Q_pi_tf_dict, self.Q_tf_dict, share_Q_pi_tf_dict = {}, {}, {}, {}

        # Actor Networks.
        with tf.variable_scope('shared_pi'):
            share_pi_tf = tf.tanh(nn(input_pi, [self.hidden] * self.share_layers))
        for i in range(self.k_heads):
            with tf.variable_scope(f'pi_{i}'):
                pi_tf = self.max_u * tf.tanh(nn(
                    share_pi_tf, [self.last_hidden] * self.layers + [self.dimu]))
            self.pi_tf_dict[i] = pi_tf
        # Critic Networks.
        with tf.variable_scope("shared_Q"):
            # for critic training
            share_Q_tf = nn(input_Q, [self.hidden] * self.share_layers)
            # for policy training
            for i in range(self.k_heads):
                input_Q_pi = tf.concat(axis=1, values=[o, g, self.pi_tf_dict[i] / self.max_u])
                share_Q_pi_tf = nn(input_Q_pi, [self.hidden] * self.share_layers, reuse=True)
                share_Q_pi_tf_dict[i] = share_Q_pi_tf
        for i in range(self.k_heads):
            with tf.variable_scope(f'Q_{i}'):
                # for critic training
                Q_tf = nn(share_Q_tf, [self.last_hidden] * self.layers + [1])
                # for policy training
                Q_pi_tf = nn(share_Q_pi_tf_dict[i], [self.last_hidden] * self.layers + [1], reuse=True)
            self.Q_tf_dict[i] = Q_tf
            self.Q_pi_tf_dict[i] = Q_pi_tf



    
    
            


    



