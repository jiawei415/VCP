import os
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from mpi4py import MPI
from threading import Thread

import vcp.algo.config as config
from vcp.common import logger
from vcp.common import tf_util
from vcp.common.util import set_global_seeds
from vcp.common.mpi_moments import mpi_moments
from vcp.algo.rollout import RolloutWorker
from vcp.algo.util import dump_params

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, random_init, all_heads_play, gen_rollout_mode, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')
        policy_path = periodic_policy_path.format('00')
        logger.info('Saving periodic policy to {} ...'.format(policy_path))
        evaluator.save_policy(policy_path)

    # random_init for o/g/rnd stat and model training
    if random_init:
        logger.info('Random initializing ...')
        rollout_worker.clear_history()
        # rollout_worker.render = True
        random_num = int(random_init) // rollout_worker.rollout_batch_size // policy.k_heads
        for epi in range(random_num):
            for head in range(policy.k_heads):
                if all_heads_play:
                    episode = rollout_worker.generate_rollouts(head, random_ac=True)
                else:
                    episode = rollout_worker.generate_rollouts(random_ac=True)
                policy.store_episode(episode, start_train=False)
        # policy.buffer.clear_buffer()

    if policy.use_her_buffer:
        push_thread = Thread(target=policy.store_in_her_buffer)
        push_thread.daemon = True
        push_thread.start()

        while True:
            if policy.her_buffer.get_current_size() > 0: break

    save_debug_log = False
    if save_debug_log:
        log_dir = logger.get_dir()
        columns = ['her/count', 'her/variance', 'no_her/count', 'no_her/variance',]
        debug_logs = pd.DataFrame(columns=columns)
        debug_logs.to_csv(os.path.join(log_dir, 'debug_log.csv'), sep="\t", index=False)

    num_rollout = 0
    best_success_rate = -1
    logger.info('Start training...')
    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        time_start = time.time()
        # train
        rollout_worker.clear_history()
        # for i in tqdm(range(n_cycles)):
        for cycle in range(n_cycles):
            for head in range(policy.k_heads):
                if all_heads_play:
                    episode = rollout_worker.generate_rollouts(head)
                else:
                    episode = rollout_worker.generate_rollouts()
                policy.store_episode(episode)
            for batch in range(n_batches):
                policy.train()
            policy.update_target_net()
        if save_debug_log and policy.use_her_buffer:
            temp_buffer = copy.deepcopy(policy.her_buffer.get_buffer())
            her_index = np.squeeze(temp_buffer['her']).astype(bool)
            count = np.squeeze(temp_buffer['count'])
            variance = np.squeeze(temp_buffer['vars'])
            her_count, her_variance = count[her_index].tolist(), variance[her_index].tolist()
            no_her_count, no_her_variance = count[~her_index].tolist(), variance[~her_index].tolist()
            debug_logs.loc[epoch] =[her_count, her_variance, no_her_count, no_her_variance]
            debug_logs.to_csv(os.path.join(log_dir, 'debug_log.csv'), sep="\t", index=False)

        # test
        evaluator.clear_history()
        evaluator.render = False
        for _ in range(n_test_rollouts):
            for head in range(policy.k_heads):
                if all_heads_play:
                    evaluator.generate_rollouts(head)
                else:
                    evaluator.generate_rollouts()

        # record logs
        time_end = time.time()
        total_time = time_end - time_start
        logger.record_tabular('epoch/num', epoch)
        logger.record_tabular('epoch/time(min)', total_time/60)
        if policy.use_her_buffer:
            logger.record_tabular('epoch/sample_produce', (policy.count - policy.last_count) * policy.mini_her_batch / total_time)
            logger.record_tabular('epoch/sample_consume', (policy.batch_size * policy.k_heads * n_batches * n_cycles) / total_time)
            policy.last_count = policy.count
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_path:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if (rank == 0 and policy_save_interval > 0 and ((epoch + 1) % policy_save_interval == 0 or epoch == 0) and save_path):
            policy_path = periodic_policy_path.format(str(epoch + 1).zfill(2))
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy


def learn(*, env, num_epoch, 
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    random_init=0,
    play_no_training=False,
    **kwargs
):

    override_params = override_params or {} 
    rank = MPI.COMM_WORLD.Get_rank()
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.spec.id

    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name.startswith('Point2D'):
        params.update(config.DEFAULT_ENV_PARAMS['Point2D'])
    if env_name.startswith('PointMass'):
        params.update(config.DEFAULT_ENV_PARAMS['PointMass'])
    elif env_name.startswith('FetchReach'):
        params.update(config.DEFAULT_ENV_PARAMS['FetchReach'])
    elif env_name.startswith('Fetch'):
        params.update(config.DEFAULT_ENV_PARAMS['Fetch'])
    elif env_name.startswith('SawyerReach'):
        params.update(config.DEFAULT_ENV_PARAMS['SawyerReach'])
    elif env_name.startswith('Hand'):
        params.update(config.DEFAULT_ENV_PARAMS['Hand'])
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter

    params.update(kwargs)   # make kwargs part of params
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs
    random_init = params['random_init']
    all_heads_play = params['all_heads_play']
    gen_rollout_mode = params['gen_rollout_mode']

    # save total params
    dump_params(logger, params)

    if rank == 0:
        config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    if load_path is not None:
        tf_util.load_variables(load_path)
    
    # no training
    if play_no_training:  
        return policy

    rollout_params = {
        'all_heads_action': params['all_heads_action_train'],
        'greedy_action': params['greedy_action_train'],
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'all_heads_action': params['all_heads_action_eval'],
        'greedy_action': params['greedy_action_eval'],
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env
    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=num_epoch, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, random_init=random_init,
        all_heads_play=all_heads_play, gen_rollout_mode=gen_rollout_mode)

