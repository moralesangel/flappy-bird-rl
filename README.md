# Flappy Bird RL

A Flappy Bird environment built to the [Gymnasium](https://gymnasium.farama.org/)
API, and a PPO agent that learns to play it. The trained policy is committed, so
the demo runs on a fresh clone without training anything.

<p align="center">
  <img src="docs/demo.gif" alt="The trained PPO agent clearing pipes" width="220">
</p>

**Trained agent: 171.7 pipes on average over 30 held-out seeds** (median 156).

```bash
pip install -r requirements.txt
python play.py            # watch the trained agent
python play.py --human    # play it yourself (SPACE)
```

---

## What is interesting here

The headline number is not the point. The useful part of this project is a
measured finding about the **observation space**, which is the kind of thing
that decides whether an agent learns at all:

> An early version of the environment exposed only the *next* pipe. The agent
> plateaued at ~13 pipes and kept dying in a specific way: falling at terminal
> velocity with the gap below it, unable to reach the opening in time.
>
> Consecutive gap centres can differ by up to 256 px, and the bird only has
> ~53 steps between pipes. Descending that far takes most of that window, so the
> agent had to start moving *before* the relevant pipe was observable. The task
> was quietly partially observable.
>
> Adding two features for the pipe after next raised held-out performance by
> roughly an order of magnitude at a *lower* step budget.

`experiments/ablation_lookahead.py` reproduces this as a controlled comparison:
identical physics, reward, network, seeds and step budget, with the two
lookahead features masked to zero in the ablated arm.

<!-- ABLATION_TABLE -->

The failure was diagnosed by logging death states (position, velocity, and which
pipe was hit) rather than by tuning hyperparameters, which is what made the cause
visible.

---

## Results

Evaluated with `python evaluate.py --episodes 30` on seeds 900–929, which are
never used during training.

| Metric | Value |
|---|---|
| Mean pipes cleared | **171.7** |
| Median | 156 |
| Min / Max | 1 / 367 |
| Episodes hitting the 20k-step cap | 5 / 30 |

Reported honestly: the maximum of 367 is the evaluation step cap, not a death,
so it is a floor rather than the policy's ceiling. Performance is also bimodal —
a few seeds still end early, which is where the remaining headroom is.

Training used PPO (`MlpPolicy`) for ~7M steps across 8 parallel environments,
about 5,500 steps/second on CPU.

---

## The environment

`FlappyBirdEnv`, registered as `FlappyBird-v0`. It passes
`gymnasium.utils.env_checker.check_env` and is deterministic given a seed.

| | |
|---|---|
| **Action** | `Discrete(2)` — 0 = do nothing, 1 = flap |
| **Observation** | `Box(7,)`, normalised to roughly `[-1, 1]` |
| **Reward** | +0.1 per step survived, +1.0 per pipe passed, −1.0 on death |
| **Termination** | Collision with a pipe, the ground, or the ceiling |
| **Truncation** | After `max_steps` (default 10,000) |

The observation is a feature vector rather than raw pixels, so an MLP policy
trains in minutes on CPU instead of hours on a GPU:

| # | Feature |
|---|---|
| 0 | Bird y position |
| 1 | Bird vertical velocity |
| 2 | Horizontal distance to the next pipe |
| 3 | Vertical offset to the top of the next gap |
| 4 | Vertical offset to the bottom of the next gap |
| 5 | Horizontal distance to the pipe after next |
| 6 | Vertical offset to the centre of the gap after next |

Feature 5 is scaled by `WIDTH + PIPE_SPACING` rather than `WIDTH`; normalising by
screen width alone left it saturated at 1.0 for most of each cycle, wasting the
input.

### Physics

Fixed timestep at 30 steps/second, so each `step()` is a reproducible unit of
time and runs are comparable across machines.

| | Value |
|---|---|
| Gravity | 0.4 px/step² |
| Flap velocity | −7.0 px/step |
| Terminal fall speed | 10.0 px/step |
| Pipe speed | 3.0 px/step |
| Pipe spacing | 160 px (≈53 steps) |
| Gap | 128 px, bird diameter 24 px → 104 px of usable band |

Collision is a circle-vs-rectangle test against each pipe half. An earlier
square-hitbox version rejected valid near-edge passes, which capped achievable
scores before any learning was involved.

---

## Repository layout

```
flappy_env.py                     the Gymnasium environment
train.py                          PPO training entry point
evaluate.py                       held-out evaluation
play.py                           interactive play, menus, game-over screen
record_demo.py                    renders docs/demo.gif
experiments/
  ablation_lookahead.py           the observation-space ablation
  ablation_results.json           its raw output
assets/                           sprites
ppo_flappy.zip                    trained policy (committed)
```

Menus and the game-over card live in `play.py`, not the environment, so
`env.step()` never emits a frame the agent does not act on.

---

## Usage

```bash
# Train from scratch (writes ppo_flappy.zip)
python train.py --timesteps 2000000

# Evaluate on held-out seeds
python evaluate.py --episodes 30

# Reproduce the ablation
python experiments/ablation_lookahead.py --timesteps 2000000 --seeds 0 1 2

# Re-record the demo GIF
python record_demo.py
```

`play.py` opens on the title card; SPACE starts and restarts, ESC quits. Pass
`--episodes N` to skip the menus for scripted runs.

Training logs to `./tb_logs`: `tensorboard --logdir tb_logs`

---

## Notes

Reward is +0.1 per step and +1.0 per pipe. The survival term is what gets
learning off the ground — with a pipe-only reward the signal is too sparse early
on, since a random policy rarely reaches the first pipe.

Sprites are from the widely circulated Flappy Bird clone asset set, used here for
a non-commercial educational project. The code is MIT licensed; the artwork is
not mine.
