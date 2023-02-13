from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from vcp.common import logger
from vcp.algo.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from vcp.algo.normalizer import Normalizer
from vcp.algo.replay_buffer import ReplayBuffer, SimpleReplayBuffer
from vcp.common.mpi_adam import MpiAdam
from vcp.common import tf_util


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, last_hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T, greedy_sample,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 her_sampler, her_bootstrap_sampler, random_sampler, priority_sampler, priority_mode, gamma, reward_fun,
                 k_heads, share_network, reverse_priority, prioritized_replay,
                 use_her_buffer, her_buffer_size, store_mode, weighted_loss, reuse=False, **kwargs):
        """Implementation of DDPG agent that is used in combination with Hindsight Experience Replay (HER).
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model. save data for her process
        stage_shapes = OrderedDict()
        stage_dtypes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
            stage_dtypes[key] = tf.float32
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
            stage_dtypes[key + '_2'] = tf.float32
        stage_shapes['r'] = (None,)
        stage_dtypes['r'] = tf.float32
        if self.use_her_buffer and self.weighted_loss:
            stage_shapes['p'] = (None,)
            stage_dtypes['p'] = tf.float32
        self.stage_shapes = stage_shapes
        self.stage_dtypes = stage_dtypes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(dtypes=list(self.stage_dtypes.values()), shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(dtype, shape=shape) for dtype, shape in zip(self.stage_dtypes.values(), self.stage_shapes.values())]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T-1 if key != 'o' else self.T, *input_shapes[key]) for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T, self.dimg)
        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size # buffer_size % rollout_batch_size should be zero
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.her_sampler, self.her_bootstrap_sampler, self.random_sampler, self.priority_sampler, self.priority_mode, store_mode='random')
        if self.use_her_buffer:
            her_buffer_shapes = {key: (1,) + val[1:] for key, val in buffer_shapes.items()}
            her_buffer_shapes['o_2'] = (1, self.dimo)
            her_buffer_shapes['ag_2'] = (1, self.dimg)
            for key in ['r', 'her', 'vars', 'mean', 'count']:
                her_buffer_shapes[key] = (1,)
            if self.prioritized_replay:
                her_buffer_shapes['p'] = (1,)
            her_buffer_size = self.her_buffer_size
            self.her_buffer = ReplayBuffer(her_buffer_shapes, her_buffer_size, 1, self.her_sampler, self.her_bootstrap_sampler, self.random_sampler, self.priority_sampler, self.priority_mode, store_mode=self.store_mode)

    def _get_reward(self, ag_2, g):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        return self.reward_fun(**reward_params)

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g, ):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs, kth_head=0):
        actions = self.get_actions(kth_head, obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None

    def action_only(self, kth_head, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.target  #self.target if use_target_net else
        action = self.sess.run(policy.pi_tf_dict[kth_head], feed_dict={
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        })
        return action

    def get_actions(self, kth_head, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf_dict[kth_head]]
        if compute_Q:
            vals += [policy.Q_pi_tf_dict[kth_head]]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def get_best_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False, greedy=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf_dict[i] for i in range(self.k_heads)]
        vals += [policy.Q_pi_tf_dict[i] for i in range(self.k_heads)]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }
        ret = self.sess.run(vals, feed_dict=feed)
        actions, Qs = ret[:self.k_heads], ret[self.k_heads:]
        if greedy:
            index = np.argmax(Qs)
        else:
            p = np.squeeze(np.exp(Qs - np.max(Qs)) / np.sum(np.exp(Qs - np.max(Qs))))
            index = np.random.choice(list(range(self.k_heads)), p=p)
        u = actions[index]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        return u

    def get_all_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        vals = self.all_actions
        if compute_Q:
            vals += self.all_Qs
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }
        ret = self.sess.run(vals, feed_dict=feed)
        actions = ret[:self.k_heads]
        u = actions[0] if self.k_heads == 1 else np.diagonal(np.array(actions)).copy().T
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u) # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        if compute_Q:
            Qs = ret[self.k_heads:]
            q = Qs[0] if self.k_heads == 1 else np.diagonal(np.array(Qs)).copy().T
            return u, q
        return u

    def get_Q(self, kth_head, o, g, u):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)

        policy = self.target  # main
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu)
        }
        ret = self.sess.run(policy.Q_tf_dict[kth_head], feed_dict=feed)
        return ret

    def get_Q_pi(self, kth_head, o, g):
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        policy = self.target
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        }
        ret = self.sess.run(policy.Q_pi_tf_dict[kth_head], feed_dict=feed)
        return ret

    def get_target_Q(self, kth_head, o, g, a, ag):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32) #??
        }

        ret = self.sess.run(policy.Q_tf_dict[kth_head], feed_dict=feed)
        return ret

    def get_mean_variance(self, o, g, u):
        policy = self.main
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu),
        }
        mean, variance = self.sess.run([self.mean, self.variance], feed_dict=feed)
        return mean, variance

    def store_episode(self, episode_batch, update_stats=True, start_train=True): #init=False
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key 'o' is of size T+1, others are of size T
        """
        self.buffer.store_episode(episode_batch)
        if update_stats:
            # episode doesn't has key o_2
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            # add transitions to normalizer
            transitions = self.her_sampler(episode_batch, num_normalizing_transitions)

            o, g, ag = transitions['o'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats
            # training normalizer online
            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()
            # if self.use_dynamic_nstep:
            self.u_stats.update(transitions['u'])
            self.u_stats.recompute_stats()

    def store_in_her_buffer(self):
        self.count, self.last_count = 0, 0
        self.mini_her_batch = self.batch_size
        ratio = 1.0 if self.only_her_data else None
        while True:
            her_epiosde_batch = self.buffer.sample(self.mini_her_batch, mode='her', ratio=ratio)
            her_epiosde_batch['mean'], her_epiosde_batch['vars'] = self.get_mean_variance(her_epiosde_batch['o'], her_epiosde_batch['g'], her_epiosde_batch['u'])
            her_epiosde_batch['count'] = np.zeros(self.mini_her_batch)
            for key in her_epiosde_batch.keys():
                her_epiosde_batch[key] = np.expand_dims(her_epiosde_batch[key], axis=1)
            self.her_buffer.store_episode(her_epiosde_batch, self.priority_temperature, self.reverse_priority, compute_priority=self.prioritized_replay)
            self.count += 1

    def _sync_optimizers(self):
        for i in range(self.k_heads):
            self.Q_optims[i].sync()
            self.pi_optims[i].sync()

    def _grads(self, kth_head):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_ops[kth_head],
            self.pi_loss_ops[kth_head],
            self.Q_grad_ops[kth_head],
            self.pi_grad_ops[kth_head],
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, kth_head, Q_grad, pi_grad):
        self.Q_optims[kth_head].update(Q_grad, self.Q_lr)
        self.pi_optims[kth_head].update(pi_grad, self.pi_lr)

    def _preprocess_transitions(self, transitions, method='list', shuffle=False):
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(transitions['r'])
            for k in transitions.keys():
                if k != 'r':
                    np.random.set_state(state)
                    np.random.shuffle(transitions[k])

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)
        if self.use_her_buffer and self.weighted_loss:
            if self.prioritized_replay:
                transitions['p'] /= np.sum(transitions['p'])
            else:
                transitions['p'] = np.ones(self.batch_size)

        if method == 'list':
            transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        else:
            transitions_batch = transitions
        return transitions_batch

    def _preprocess_ag(self, future_ags, transition_her):
        vars = []
        for future_ag in future_ags:
            vars.append(self.get_variance(transition_her['o'], future_ag, transition_her['u']))
        vars = np.vstack(vars)
        future_ag = np.stack(future_ags, axis=0)
        indexs = np.argmin(vars, axis=0) if self.reverse_priority else np.argmax(vars, axis=0)
        her_vars = vars[indexs, np.arange(future_ag.shape[1])]
        her_ag = future_ag[indexs, np.arange(future_ag.shape[1])]
        transition_her['g'] = her_ag
        return transition_her, her_vars

    def sample_batch(self):
        if self.use_her_buffer:
            ratio, shuffle = (0.8, True) if self.sample_from_buffer else (1.0, False)
            mode = 'priority' if self.prioritized_replay else 'random'
            her_batch_size = int(self.batch_size * ratio)
            orignal_batch_size = self.batch_size - her_batch_size
            transition_her = self.her_buffer.sample(her_batch_size, mode=mode, greedy=self.greedy_sample)
            if self.sample_from_buffer:
                transition_orignal = self.buffer.sample(orignal_batch_size, mode='random')
                transitions = {k: np.concatenate([transition_her[k], transition_orignal[k]], axis=0) for k in transition_orignal.keys()}
            else:
                transitions = transition_her
            if 'r' not in transitions.keys(): transitions['r'] = self._get_reward(transitions['ag_2'], transitions['g'])
            transitions = self._preprocess_transitions(transitions, shuffle=shuffle)
        else:
            ratio = None if self.sample_from_buffer else 1.0
            transitions = self.buffer.sample(self.batch_size, mode='her', ratio=ratio)
            transitions = self._preprocess_transitions(transitions, shuffle=False)
        return transitions

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
            self.temp_batch = batch
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        for i in range(self.k_heads):
            if stage:
                self.stage_batch()
            critic_loss, actor_loss, _, _ = self.sess.run([
                self.Q_loss_ops[i],
                self.pi_loss_ops[i],
                self.Q_train_ops[i],
                self.pi_train_ops[i],
            ])
            self.critic_loss_dict[i].append(critic_loss)
            self.actor_loss_dict[i].append(actor_loss)

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        for i in range(self.k_heads):
            self.sess.run(self.update_target_net_ops[i])

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        # assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()
        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('u_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.u_stats = Normalizer(self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        if self.use_her_buffer and self.weighted_loss:
            weight_tf = tf.reshape(batch_tf['p'], [-1, 1])

        # network
        with tf.variable_scope(f'main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope(f'target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()

        self.log_op_list = [self.o_stats.mean, self.o_stats.std, self.g_stats.mean, self.g_stats.std, self.u_stats.mean, self.u_stats.std]
        self.all_actions = [self.main.pi_tf_dict[i] for i in range(self.k_heads)]
        self.all_Qs = [self.main.Q_tf_dict[i] for i in range(self.k_heads)]
        # for coumpute variance of Q value
        Qs = tf.concat(axis=1, values=self.all_Qs)
        self.mean, self.variance = tf.nn.moments(Qs, axes=1)

        # build ops for train
        self.Q_loss_ops, self.pi_loss_ops = {}, {}
        self.Q_train_ops, self.pi_train_ops = {}, {}
        self.update_target_net_ops = {}
        for i in range(self.k_heads):
            # critic loss
            target_Q_pi_tf = self.target.Q_pi_tf_dict[i]
            clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
            target_Q_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
            td_error_tf = tf.square(tf.stop_gradient(target_Q_tf) - self.main.Q_tf_dict[i])
            if self.use_her_buffer and self.weighted_loss:
                td_error_tf = tf.multiply(td_error_tf, weight_tf)
            Q_loss_tf = tf.reduce_mean(td_error_tf)
            self.Q_loss_ops[i] = Q_loss_tf
            # actor_loss
            main_Q_pi_tf = self.main.Q_pi_tf_dict[i]
            pi_reg_tf = tf.square(self.main.pi_tf_dict[i] / self.max_u)
            if self.use_her_buffer and self.weighted_loss:
                main_Q_pi_tf = tf.multiply(main_Q_pi_tf, weight_tf)
                pi_reg_tf = tf.multiply(pi_reg_tf, weight_tf)
            pi_loss_tf = -tf.reduce_mean(main_Q_pi_tf) + self.action_l2 * tf.reduce_mean(pi_reg_tf)
            self.pi_loss_ops[i] = pi_loss_tf
            # update main net ops
            main_Q_vars = self._vars('main/shared_Q') + self._vars(f'main/Q_{i}/')
            main_pi_vars = self._vars('main/shared_pi') + self._vars(f'main/pi_{i}/')
            self.Q_train_ops[i] = tf.train.AdamOptimizer(self.Q_lr).minimize(Q_loss_tf, var_list=main_Q_vars)
            self.pi_train_ops[i] = tf.train.AdamOptimizer(self.pi_lr).minimize(pi_loss_tf, var_list=main_pi_vars)
            # update target net ops
            main_vars = main_Q_vars + main_pi_vars
            target_vars = self._vars(f'target/shared_Q') + self._vars(f'target/Q_{i}/') + self._vars('target/shared_pi') + self._vars(f'target/pi_{i}/')
            self.update_target_net_ops[i] = list(
                map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(target_vars, main_vars))) # polyak averaging

        main_vars, target_vars = self._vars('main'), self._vars('target')
        self.init_target_net_op = list(map(lambda v: v[0].assign(v[1]), zip(target_vars, main_vars)))

        self.sess.run(tf.variables_initializer(self._global_vars(""))) # init global vars
        self._init_target_net()
        self.critic_loss_dict, self.actor_loss_dict = {k: [] for k in range(self.k_heads)}, {k: [] for k in range(self.k_heads)}

    def logs(self, prefix=''):
        o_mean, o_std, g_mean, g_std, u_mean, u_std = self.sess.run(self.log_op_list)
        logs = []
        logs += [('stats/o_mean', np.mean(o_mean)), ('stats/o_std', np.mean(o_std))]
        logs += [('stats/g_mean', np.mean(g_mean)), ('stats/g_std', np.mean(g_std))]
        logs += [('stats/u_mean', np.mean(u_mean)), ('stats/u_std', np.mean(u_std))]
        for i, (critic_loss, actor_loss) in enumerate(zip(self.critic_loss_dict.values(), self.actor_loss_dict.values())):
            logs += [(f'head/head_{i}_critic_loss', np.mean(critic_loss))]
            logs += [(f'head/head_{i}_actor_loss', np.mean(actor_loss))]
            self.critic_loss_dict[i], self.actor_loss_dict[i]  = [], []
        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)

    def save(self, save_path):
        tf_util.save_variables(save_path)

