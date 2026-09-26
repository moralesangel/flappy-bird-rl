"""Ablation: does seeing the pipe after next matter?

The full environment exposes seven features; the "no-lookahead" arm masks the
last two (distance and gap offset of the pipe after next) to zero, leaving the
agent with only the pipe immediately ahead. Everything else -- physics, reward,
network, seeds, step budget -- is held fixed.

Usage:  python experiments/ablation_lookahead.py --timesteps 2000000
"""

import argparse
import json
import os
import statistics
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flappy_env import FlappyBirdEnv  # noqa: E402

EVAL_SEEDS = list(range(900, 930))  # held out from every training seed


class NoLookahead(gym.ObservationWrapper):
    """Zero out the two features describing the pipe after next."""

    def observation(self, obs):
        obs = obs.copy()
        obs[5:] = 0.0
        return obs


def make_env(lookahead):
    def _init():
        env = FlappyBirdEnv()
        return env if lookahead else NoLookahead(env)

    return _init


def evaluate(model, lookahead, max_steps=20_000):
    env = FlappyBirdEnv(max_steps=max_steps)
    wrapped = env if lookahead else NoLookahead(env)
    scores = []
    for seed in EVAL_SEEDS:
        obs, _ = wrapped.reset(seed=seed)
        done = False
        info = {"score": 0}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = wrapped.step(action)
            done = terminated or truncated
        scores.append(info["score"])
    return scores


def run_arm(name, lookahead, timesteps, seed):
    venv = make_vec_env(make_env(lookahead), n_envs=8, seed=seed)
    model = PPO(
        "MlpPolicy", venv, learning_rate=3e-4, n_steps=1024, batch_size=256,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01, verbose=0, seed=seed,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    scores = evaluate(model, lookahead)
    venv.close()
    return {
        "arm": name,
        "seed": seed,
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "max": max(scores),
        "min": min(scores),
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default="experiments/ablation_results.json")
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        for name, lookahead in [("with_lookahead", True), ("no_lookahead", False)]:
            r = run_arm(name, lookahead, args.timesteps, seed)
            print(
                f"{r['arm']:15s} seed={seed}  mean={r['mean']:7.1f}  "
                f"median={r['median']:6.1f}  max={r['max']:5d}",
                flush=True,
            )
            results.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"timesteps": args.timesteps, "eval_seeds": EVAL_SEEDS, "runs": results}, f, indent=2)

    print(f"\nwrote {args.out}\n")
    for name in ("with_lookahead", "no_lookahead"):
        arm = [r["mean"] for r in results if r["arm"] == name]
        print(f"{name:15s} mean over {len(arm)} seeds: {statistics.mean(arm):7.1f}")


if __name__ == "__main__":
    main()
