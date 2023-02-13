from vcp.common import logger
import numpy as np

def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions): 
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the u and o_2 ag_2
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        if 'r' not in transitions.keys():
            transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    return _random_sample


def make_default_sample(reward_fun):
    def _get_reward(transitions):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the goal
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        return reward_fun(**reward_params)

    def _reshape_transitions(transitions, batch_size):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size)
        return transitions

    def _random_sample(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}
        if 'r' not in transitions.keys():
            transitions['r'] = _get_reward(transitions)
        return _reshape_transitions(transitions, batch_size), episode_idxs

    def _priority_sample(episode_batch, batch_size_in_transitions, greedy=False):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        # Select which episodes and time steps to use.
        p = np.squeeze(episode_batch['p'].copy())
        try:
            if greedy:
                episode_idxs = np.argpartition(p, -batch_size)[-batch_size:]
            else:
                episode_idxs = np.random.choice(np.arange(rollout_batch_size), batch_size, p=p)
        except ValueError:
            logger.error(f"priority sampler error!")
            p /= p.sum()
            episode_idxs = np.random.choice(np.arange(rollout_batch_size), batch_size, p=p)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}
        if 'r' not in transitions.keys():
            transitions['r'] = _get_reward(transitions)
        return _reshape_transitions(transitions, batch_size), episode_idxs
    return _random_sample, _priority_sample


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, no_her=False):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  
        future_p = 0

    if no_her:
        print( '*' * 10 + 'Do not use HER in this method' + '*' * 10)

    def _preprocess(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout

        # Select which episodes and time steps to use. 
        # np.random.randint doesn't contain the last one, so comes from 0 to rollout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, ratio):
        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = (np.random.uniform(size=batch_size) < ratio)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)  
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        return future_ag.copy(), her_indexes.copy()

    def _get_reward(ag_2, g):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        return reward_fun(**reward_params)

    def _reshape_transitions(transitions, batch_size):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size)
        return transitions

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, ratio=None, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        if not no_her:
            if ratio is None: ratio = future_p
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, ratio)
            transitions['g'][her_indexes] = future_ag
            transitions['her'] = her_indexes
        if 'r' not in transitions.keys():
            transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size)

    def _sample_bootstrap_her_transitions(episode_batch, batch_size_in_transitions, k_goals, ratio, info=None):
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        if ratio is None: ratio = future_p
        her_indexes = (np.random.uniform(size=batch_size) < ratio)
        future_ag_list = []
        for _ in range(k_goals):
            future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
            future_offset = future_offset.astype(int)
            future_t = (t_samples + 1 + future_offset)[her_indexes]
            future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
            future_ag_list.append(future_ag.copy())
        transition_her = {k: transitions[k][her_indexes] for k in transitions.keys()}
        transition_orignal = {k: transitions[k][~her_indexes] for k in transitions.keys()}
        return future_ag_list, transition_her, transition_orignal

    return _sample_her_transitions, _sample_bootstrap_her_transitions
