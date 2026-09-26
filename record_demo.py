"""Record a GIF of the trained agent playing, for the README.

Usage:  python record_demo.py --frames 420 --out docs/demo.gif
"""

import argparse
import os

from PIL import Image
from stable_baselines3 import PPO

from flappy_env import FlappyBirdEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ppo_flappy")
    parser.add_argument("--out", default="docs/demo.gif")
    parser.add_argument("--frames", type=int, default=420)
    parser.add_argument("--seed", type=int, default=901)
    parser.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    parser.add_argument("--scale", type=float, default=0.75)
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    env = FlappyBirdEnv(render_mode="rgb_array", max_steps=args.frames + 10)
    obs, _ = env.reset(seed=args.seed)

    frames = []
    for i in range(args.frames):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if i % args.stride == 0:
            img = Image.fromarray(env.render())
            if args.scale != 1.0:
                img = img.resize(
                    (int(img.width * args.scale), int(img.height * args.scale)),
                    Image.NEAREST,  # keep the pixel art crisp
                )
            frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))
        if terminated or truncated:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / 30 * args.stride),
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}: {len(frames)} frames, {size_kb:.0f} KB, score reached {info['score']}")


if __name__ == "__main__":
    main()
