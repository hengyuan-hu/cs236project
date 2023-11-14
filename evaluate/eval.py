import pickle
import torch
import numpy as np
from robosuite_wrapper import PixelRobosuite
from common_utils import Recorder
from common_utils import ibrl_utils as utils


def run_eval(
    env: PixelRobosuite,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    save_frame=False,
    eval_mode=True,
):
    recorder = None if record_dir is None else Recorder(record_dir)

    scores = []
    lens = []
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            obs, image_obs = env.reset()

            try:
                agent.reset()
            except AttributeError:
                pass

            terminal = False
            while not terminal:
                if recorder is not None:
                    recorder.add(image_obs)

                action = agent.act(obs, eval_mode=eval_mode).numpy()
                obs, reward, terminal, _, image_obs = env.step(action)
                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))
            if scores[-1] > 0:
                lens.append(env.time_step)

            if recorder is not None:
                save_path = recorder.save(f"episode{episode_idx}", save_frame)
                reward_path = f"{save_path}.reward.pkl"
                print(f"saving reward to {reward_path}")
                pickle.dump(rewards, open(reward_path, "wb"))

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")
        print(f"average time for success games: {np.mean(lens)}")

    return scores


if __name__ == "__main__":
    import argparse
    import os
    import time
    import common_utils
    import train_bc
    from multi_process_eval import run_eval as mp_run_eval

    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bc", help="bc/rl")
    parser.add_argument("--num_game", type=int, default=10)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--save_frame", type=int, default=0)
    parser.add_argument("--mp", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    common_utils.set_all_seeds(args.seed)

    if args.folder is None:
        agent, env, env_params = train_bc.load_model(args.weight, "cuda")
        eval_items = [(args.weight, agent, env, env_params)]
    else:
        weights = common_utils.get_all_files(args.folder, ".pt")
        eval_items = []
        for weight in weights:
            agent, env, env_params = train_bc.load_model(weight, "cuda")
            eval_items.append((weight, agent, env, env_params))

    weight_scores = []
    for weight, agent, env, env_params in eval_items:
        t = time.time()
        if args.mp >= 1:
            assert not args.save_frame
            assert args.record_dir is None
            scores = mp_run_eval(
                env_params, agent, args.num_game, args.mp, args.seed, verbose=args.verbose
            )
        else:
            scores = run_eval(
                env,
                agent,
                args.num_game,
                args.seed,
                args.record_dir,
                save_frame=args.save_frame,
                verbose=args.verbose,
            )
        print(f"weight: {weight}")
        print(f"score: {np.mean(scores)}, time: {time.time() - t:.1f}")
        weight_scores.append((weight, np.mean(scores)))

    if len(weight_scores) > 1:
        weight_scores = sorted(weight_scores, key=lambda x: -x[1])
        scores = []
        for weight, score in weight_scores:
            print(weight, score)
            scores.append(score)
        print(f"average score: {100 * np.mean(scores):.2f}")
        print(f"max score: {100 * np.mean(scores[0]):.2f}")
