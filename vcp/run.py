import os
import sys
import numpy as np

from vcp.common.env_util import get_env_type, build_env, get_game_envs
from vcp.common.parse_args import get_learn_function_defaults, parse_cmdline_kwargs, common_arg_parser
from vcp.common import logger
from vcp.algo.train import learn
from vcp.util import init_logger


_game_envs = get_game_envs(print_out=False)

def train(args, extra_args):
    env_type, env_id = get_env_type(args, _game_envs)
    print('env_type: {}'.format(env_type))
    seed = args.seed
    alg_kwargs = get_learn_function_defaults('her', env_type)
    alg_kwargs.update(extra_args)
    alg_kwargs.update(eval(args.alg_config))
    env = build_env(args, _game_envs)
    print('Training {}:{} with arguments \n{}'.format(env_type, env_id, alg_kwargs))

    ## make save dir
    if args.save_path:
        args.save_path = os.path.join(logger.get_dir(), args.save_path)
        os.makedirs(os.path.expanduser(args.save_path), exist_ok=True)

    model = learn(
        env=env,
        seed=seed,
        num_epoch=args.num_epoch,
        save_path=args.save_path,
        load_path=args.load_path,
        play_no_training=args.play_no_training,
        **alg_kwargs
    )
    return model, env


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    rank = init_logger(args)

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = os.path.expanduser(args.save_path)
        last_policy_path = os.path.join(save_path, 'policy_last.pkl')
        model.save(last_policy_path)

    if args.play or args.play_no_training:
        logger.log("Running trained model")
        state = model.initial_state if hasattr(model, 'initial_state') else None
        num_step, num_episode= 50, 100
        num_success = 0
        for episode in range(num_episode):
            episode_rew = np.zeros(args.num_env)
            dones = np.zeros((1,))
            obs = env.reset()
            for step in range(num_step):
                if state is not None:
                    actions, _, state, _ = model.step(obs, S=state, M=dones)
                else:
                    actions, _, _, _ = model.step(obs, kth_head=0)
                obs, rew, done, info = env.step(actions)
                # env.render()
                episode_rew += rew
                success = np.array([i.get('is_success', 0.0) for i in info])
                if any(done) or any(success):
                    print(f"episode: {episode} reward: {episode_rew}, step: {step}, success: {success}")
                    if any(success): num_success += 1
                    break
        print(f"success rate: {num_success/num_episode}")

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
