"""Evaluate a trained policy on held-out seeds.

Usage:  python evaluate.py --episodes 30
"""

import argparse
import statistics

from stable_baselines3 import PPO

from flappy_env import FlappyBirdEnv

# Seeds 900+ are never used for training, so these numbers are held out.
EVAL_SEED_BASE = 900


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ppo_flappy")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    env = FlappyBirdEnv(max_steps=args.max_steps)

    scores, truncated_count = [], 0
    for i in range(args.episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        done = False
        info = {"score": 0}
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        truncated_count += bool(truncated)
        scores.append(info["score"])

    scores_sorted = sorted(scores)
    print(f"episodes      : {len(scores)} (seeds {EVAL_SEED_BASE}..{EVAL_SEED_BASE + len(scores) - 1})")
    print(f"mean pipes    : {statistics.mean(scores):.1f}")
    print(f"median pipes  : {statistics.median(scores):.0f}")
    print(f"min / max     : {min(scores)} / {max(scores)}")
    print(f"hit step cap  : {truncated_count}/{len(scores)} (max_steps={args.max_steps})")
    print(f"scores        : {scores_sorted}")


if __name__ == "__main__":
    main()
