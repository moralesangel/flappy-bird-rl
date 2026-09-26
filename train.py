"""Train a PPO agent on FlappyBird-v0."""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from flappy_env import FlappyBirdEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--model-path", default="ppo_flappy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_vec_env(FlappyBirdEnv, n_envs=args.n_envs, seed=args.seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log="./tb_logs",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(args.model_path)
    print(f"saved -> {args.model_path}.zip")

    # Quick greedy evaluation.
    eval_env = Monitor(FlappyBirdEnv())
    scores = []
    for _ in range(10):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        scores.append(info["score"])
    print(f"eval scores over 10 episodes: {scores}  mean={sum(scores) / len(scores):.1f}")


if __name__ == "__main__":
    main()
