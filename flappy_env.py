"""Gymnasium environment for the Flappy Bird game in game.py."""

import os
import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

WIDTH, HEIGHT = 288, 512

BIRD_RADIUS = 12  # matches the 34x24 bird sprite (half its height)
BIRD_X = WIDTH // 3

# Physics, per fixed simulation step (the game runs at FPS steps per second).
FPS = 30
GRAVITY = 0.4
JUMP_VELOCITY = -7.0
MAX_FALL_SPEED = 10.0

PIPE_WIDTH = 52  # the real width of assets/pipe-green.png
PIPE_SPEED = 3.0
PIPE_SPACING = 160  # horizontal distance between consecutive pipes
HOLE_SIZE = 0.25  # fraction of screen height that is the gap
HOLE_MIN, HOLE_MAX = 0.15, 0.65  # range for the top of the gap

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

background = pygame.image.load(os.path.join(_ASSETS, "background-day.png"))

# The pipe sprite is 52x320, but a pipe has to cover up to HEIGHT px of screen,
# so stretch it vertically once here rather than on every frame. The top pipe is
# the same image flipped, which keeps its flared lip facing the gap.
_pipe_src = pygame.image.load(os.path.join(_ASSETS, "pipe-green.png"))
pipe_image = pygame.transform.scale(_pipe_src, (PIPE_WIDTH, HEIGHT))
pipe_image_top = pygame.transform.flip(pipe_image, False, True)

bird_image = pygame.image.load(os.path.join(_ASSETS, "yellowbird-midflap.png"))

# Digits are not all the same width ("1" is 16px, the rest 24), so the score is
# laid out by each sprite's own width rather than a fixed stride.
digit_images = [pygame.image.load(os.path.join(_ASSETS, f"{d}.png")) for d in range(10)]

# Title card for the menu and the "game over" banner, both used by play.py.
message_image = pygame.image.load(os.path.join(_ASSETS, "message.png"))
gameover_image = pygame.image.load(os.path.join(_ASSETS, "gameover.png"))


def draw_score(screen, score, centre_x=WIDTH // 2, top=20, scale=1.0):
    """Blit `score` as digit sprites, horizontally centred on centre_x.

    `scale` shrinks or grows the digits, which the game-over card uses to tell
    the final score apart from the best-so-far without extra label assets.
    """
    digits = [digit_images[int(c)] for c in str(int(score))]
    if scale != 1.0:
        digits = [
            pygame.transform.scale(
                d, (max(1, int(d.get_width() * scale)), max(1, int(d.get_height() * scale)))
            )
            for d in digits
        ]
    total = sum(d.get_width() for d in digits)
    x = centre_x - total // 2
    for d in digits:
        screen.blit(d, (x, top))
        x += d.get_width()


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.v = 0.0

    def move(self):
        self.v = min(self.v + GRAVITY, MAX_FALL_SPEED)
        self.y += self.v

    def jump(self):
        self.v = JUMP_VELOCITY

    def draw(self, screen):
        rect = bird_image.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(bird_image, rect)


class Pipe:
    def __init__(self, x, rng):
        self.x = float(x)
        self.width = PIPE_WIDTH
        self.hole_size = HOLE_SIZE
        self.hole = rng.uniform(HOLE_MIN, HOLE_MAX)
        self.scored = False

    @property
    def gap_top(self):
        return self.hole * HEIGHT

    @property
    def gap_bottom(self):
        return (self.hole + self.hole_size) * HEIGHT

    def move(self, speed):
        self.x -= speed

    def recycle(self, x, rng):
        self.x = float(x)
        self.hole = rng.uniform(HOLE_MIN, HOLE_MAX)
        self.scored = False

    def _hits_rect(self, bird, top, bottom):
        """Circle-vs-rectangle: true distance from the bird's centre to the rect."""
        nearest_x = max(self.x, min(bird.x, self.x + self.width))
        nearest_y = max(top, min(bird.y, bottom))
        dx, dy = bird.x - nearest_x, bird.y - nearest_y
        return dx * dx + dy * dy < BIRD_RADIUS * BIRD_RADIUS

    def collides_with(self, bird):
        return self._hits_rect(bird, 0, self.gap_top) or self._hits_rect(
            bird, self.gap_bottom, HEIGHT
        )

    def draw(self, screen):
        # Both sprites are already HEIGHT tall, so anchoring the top one at
        # gap_top - HEIGHT puts its (flipped) mouth exactly on the gap edge.
        screen.blit(pipe_image_top, (self.x, self.gap_top - HEIGHT))
        screen.blit(pipe_image, (self.x, self.gap_bottom))



class FlappyBirdEnv(gym.Env):
    """Flappy Bird as a Gymnasium environment.

    Action space: Discrete(2) -- 0 = do nothing, 1 = flap.

    Observation space: Box(7,), all normalised to roughly [-1, 1]:
        0. bird y position               (0 = top, 1 = bottom)
        1. bird vertical velocity        (scaled by MAX_FALL_SPEED)
        2. horizontal distance to the next pipe
        3. vertical offset to the top of the next gap
        4. vertical offset to the bottom of the next gap
        5. horizontal distance to the pipe after next
        6. vertical offset to the centre of the gap after next

    The pipe after next matters: consecutive gap centres can differ by up to
    (HOLE_MAX - HOLE_MIN) * HEIGHT px, which takes most of the travel time
    between pipes to cross. Without it the agent cannot start moving in time.

    Reward: +0.1 per step survived, +1.0 for passing a pipe, -1.0 on death.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None, max_steps=10_000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.screen = None
        self.clock = None

        self.bird = None
        self.pipes = []
        self.score = 0
        self.steps = 0
        self._rng = random.Random()

    # ---------------------------------------------------------------- helpers

    def _upcoming_pipes(self):
        """The pipes ahead of the bird, nearest first."""
        if self.bird is None:
            raise RuntimeError("Environment must be reset before observations are requested")
        ahead = sorted(
            (p for p in self.pipes if p.x + p.width >= self.bird.x - BIRD_RADIUS),
            key=lambda p: p.x,
        )
        # Late in a recycle step every pipe can briefly sit behind the bird.
        return ahead if ahead else sorted(self.pipes, key=lambda p: p.x)

    def _next_pipe(self):
        """The first pipe the bird has not yet cleared."""
        return self._upcoming_pipes()[0]

    def _get_obs(self):
        ahead = self._upcoming_pipes()  # raises if the env was never reset
        assert self.bird is not None
        pipe = ahead[0]
        # Fall back to the same pipe when only one is ahead, so the observation
        # keeps its shape without inventing a gap position.
        after = ahead[1] if len(ahead) > 1 else pipe

        dx = (pipe.x + pipe.width - self.bird.x) / WIDTH
        # The pipe after next can be over a screen width away, so scale it by the
        # distance it actually spans or the feature would sit pinned at 1.0.
        dx2 = (after.x + after.width - self.bird.x) / (WIDTH + PIPE_SPACING)
        after_centre = (after.gap_top + after.gap_bottom) / 2
        return np.array(
            [
                self.bird.y / HEIGHT,
                self.bird.v / MAX_FALL_SPEED,
                np.clip(dx, 0.0, 1.0),
                (self.bird.y - pipe.gap_top) / HEIGHT,
                (self.bird.y - pipe.gap_bottom) / HEIGHT,
                np.clip(dx2, 0.0, 1.0),
                np.clip((self.bird.y - after_centre) / HEIGHT, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # ------------------------------------------------------------- gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng.seed(seed)

        self.bird = Bird(BIRD_X, HEIGHT / 2)
        n_pipes = int(WIDTH / PIPE_SPACING) + 2
        self.pipes = [Pipe(WIDTH + i * PIPE_SPACING, self._rng) for i in range(n_pipes)]
        self.score = 0
        self.steps = 0

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.bird is None:
            raise RuntimeError("Environment must be reset before steps are taken")
        if action == 1:
            self.bird.jump()
        self.bird.move()

        rightmost = max(p.x for p in self.pipes)
        for pipe in self.pipes:
            pipe.move(PIPE_SPEED)
            if pipe.x + pipe.width < 0:
                pipe.recycle(rightmost + PIPE_SPACING, self._rng)

        reward = 0.1  # staying alive
        if self.bird is None:
            raise RuntimeError("Environment must be reset before steps are taken")
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + pipe.width < self.bird.x - BIRD_RADIUS:
                pipe.scored = True
                self.score += 1
                reward += 1.0

        terminated = False
        hit_ground = self.bird.y + BIRD_RADIUS >= HEIGHT
        hit_ceiling = self.bird.y - BIRD_RADIUS <= 0
        if hit_ground or hit_ceiling or any(p.collides_with(self.bird) for p in self.pipes):
            terminated = True
            reward = -1.0

        self.steps += 1
        truncated = self.steps >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # -------------------------------------------------------------- rendering

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def ensure_surface(self):
        """Create the window (or offscreen surface) if it does not exist yet."""
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Flappy Bird RL")
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((WIDTH, HEIGHT))
        return self.screen

    def draw_scene(self):
        """Paint the current world onto the surface without advancing anything.

        Menus and the game-over card reuse this to keep the frozen game visible
        behind their overlay.
        """
        if self.bird is None:
            raise RuntimeError("Environment must be reset before rendering")
        screen = self.ensure_surface()
        screen.fill((0, 100, 200))
        screen.blit(background, (0, 0))
        for pipe in self.pipes:
            pipe.draw(screen)
        self.bird.draw(screen)
        draw_score(screen, self.score)

    def _render_frame(self):
        if self.bird is None:
            raise RuntimeError("Environment must be reset before observations are requested")
        self.ensure_surface()
        assert self.screen is not None

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit

        self.draw_scene()

        if self.render_mode == "human":
            clock = self.clock
            if clock is None:
                clock = pygame.time.Clock()
                self.clock = clock
            pygame.display.flip()
            clock.tick(FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


gym.register(id="FlappyBird-v0", entry_point="flappy_env:FlappyBirdEnv")
