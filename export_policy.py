"""Export the trained actor network to JSON for the browser demo.

Only the policy (actor) path is needed for inference: the value head and the
feature extractors are dropped. The result is a few thousand floats, which is
small enough to ship as plain JSON and evaluate in JavaScript without any ML
runtime.

Usage:  python export_policy.py --out docs/policy.json
"""

import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from flappy_env import (
    BIRD_RADIUS,
    FPS,
    GRAVITY,
    HEIGHT,
    HOLE_MAX,
    HOLE_MIN,
    HOLE_SIZE,
    JUMP_VELOCITY,
    MAX_FALL_SPEED,
    PIPE_SPACING,
    PIPE_SPEED,
    PIPE_WIDTH,
    WIDTH,
    BIRD_X,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ppo_flappy")
    parser.add_argument("--out", default="docs/policy.json")
    args = parser.parse_args()

    model = PPO.load(args.model_path)
    sd = {k: v.cpu().numpy() for k, v in model.policy.state_dict().items()}

    # Linear layers are stored as (out, in); the JS side multiplies as
    # y[o] = sum_i W[o][i] * x[i] + b[o], so keep that orientation.
    layers = [
        {"W": sd["mlp_extractor.policy_net.0.weight"], "b": sd["mlp_extractor.policy_net.0.bias"], "act": "tanh"},
        {"W": sd["mlp_extractor.policy_net.2.weight"], "b": sd["mlp_extractor.policy_net.2.bias"], "act": "tanh"},
        {"W": sd["action_net.weight"], "b": sd["action_net.bias"], "act": "none"},
    ]

    payload = {
        "note": "PPO actor for FlappyBird-v0. argmax over the 2 logits = deterministic action.",
        "obs_dim": 7,
        "n_actions": 2,
        "layers": [
            {"W": [[round(float(x), 7) for x in row] for row in L["W"]],
             "b": [round(float(x), 7) for x in L["b"]],
             "act": L["act"]}
            for L in layers
        ],
        # The browser re-implements the environment, so ship the constants with
        # the weights; a mismatch here silently changes the policy's behaviour.
        "env": {
            "WIDTH": WIDTH, "HEIGHT": HEIGHT, "FPS": FPS,
            "GRAVITY": GRAVITY, "JUMP_VELOCITY": JUMP_VELOCITY,
            "MAX_FALL_SPEED": MAX_FALL_SPEED,
            "PIPE_WIDTH": PIPE_WIDTH, "PIPE_SPEED": PIPE_SPEED,
            "PIPE_SPACING": PIPE_SPACING,
            "HOLE_SIZE": HOLE_SIZE, "HOLE_MIN": HOLE_MIN, "HOLE_MAX": HOLE_MAX,
            "BIRD_RADIUS": BIRD_RADIUS, "BIRD_X": BIRD_X,
        },
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    n = sum(np.prod(L["W"].shape) + L["b"].shape[0] for L in layers)
    import os
    print(f"wrote {args.out}: {n} parameters, {os.path.getsize(args.out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
