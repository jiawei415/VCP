from collections import deque

import numpy as np

from vcp.algo.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, monitor=False,
                 all_heads_action=False, greedy_action=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.
        Args:
            venv: vectorized gym environments.
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        assert self.T > 0
        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.episode_len_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()

    def generate_mask(self):
        if self.policy.k_heads == 1:
            mask = np.ones(shape=(self.rollout_batch_size, self.policy.k_heads)).astype(np.float32)
        else:
            mask = np.random.binomial(1, 0.5, size=(self.rollout_batch_size, self.policy.k_heads)).astype(np.float32)
        return mask

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_ag = self.obs_dict['achieved_goal']
        self.initial_mask = self.generate_mask()
        self.g = self.obs_dict['desired_goal']

    def generate_rollouts(self, kth_head=0, random_ac=False):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()
        q_mask = np.ones(shape=(self.rollout_batch_size, self.policy.k_heads)).astype(np.int32)
        if not random_ac:
            q_mask[:, kth_head] = 0

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag
        mask = np.empty((self.rollout_batch_size, self.policy.k_heads), np.float32)
        mask[:] = self.initial_mask

        # generate episodes
        masks, q_masks = [], []
        obs, achieved_goals, acts, goals, successes, rewards = [], [], [], [], [], []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        episode_len = 0
        for t in range(self.T):
            episode_len += 1
            if random_ac:
                u = self.policy._random_action(self.rollout_batch_size)
            else:
                policy_output = self.policy.get_actions(
                    kth_head,
                    o, ag, self.g,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net,
                    compute_Q=self.compute_Q,
                )
                if self.compute_Q:
                    u, Q = policy_output
                    Qs.append(Q)
                else:
                    u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            obs_dict_new, reward, done, info = self.venv.step(u)
            if self.render:
                self.venv.render()
            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']
            mask_new = self.generate_mask()
            success = np.array([i.get('is_success', 0.0) for i in info])

            if any(done) or t == self.T-1:
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                break

            if len(self.info_keys) > 0:
                for i, info_dict in enumerate(info):
                    for idx, key in enumerate(self.info_keys):
                        try:
                            info_values[idx][t, i] = info[i][key]
                        except:
                            pass

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            dones.append(done)
            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            rewards.append(reward.copy())
            masks.append(mask.copy())
            q_masks.append(q_mask.copy())
            o[...] = o_new
            ag[...] = ag_new
            mask[...] = mask_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        masks.append(mask.copy())
        q_masks.append(q_mask.copy())

        episode = dict(
            o=obs,
            u=acts,
            g=goals,
            ag=achieved_goals,
            # r=rewards,
            # m=masks,
            # q_m=q_masks,
        )
        if len(self.info_keys) > 0:
            for key, value in zip(self.info_keys, info_values):
                episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        self.reward_history.append(np.sum(rewards))
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.episode_len_history.append(episode_len)
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)  # change shape to (rollout, steps, dim)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()
        self.episode_len_history.clear()
        self.reward_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        self.policy.save(path)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('total_reward', np.mean(self.reward_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode_num', self.n_episodes)]
        logs += [('episode_len', np.mean(self.episode_len_history))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs