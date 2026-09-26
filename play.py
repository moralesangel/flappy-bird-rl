"""Watch a trained agent play, or play yourself with SPACE."""

import argparse

import pygame

from flappy_env import (
    FPS,
    HEIGHT,
    WIDTH,
    FlappyBirdEnv,
    draw_score,
    gameover_image,
    message_image,
)

# Menus and the game-over card live here rather than in the environment: they are
# presentation states, not part of the MDP, and stepping through them during
# training would feed the agent frames it never acts on.


def _pump():
    """Handle window events; returns False if the player closed the window."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


def _wait_for_key(env, draw_overlay, clock):
    """Block on an overlay until SPACE is pressed. False if the player quit."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    return True
        draw_overlay()
        pygame.display.flip()
        clock.tick(FPS)


def _blit_centred(screen, image, y):
    screen.blit(image, image.get_rect(center=(WIDTH // 2, y)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ppo_flappy")
    parser.add_argument("--human", action="store_true", help="control the bird with SPACE")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="run N episodes without menus (default: interactive)",
    )
    args = parser.parse_args()

    model = None
    if not args.human:
        import os

        from stable_baselines3 import PPO

        path = args.model_path if args.model_path.endswith(".zip") else args.model_path + ".zip"
        if not os.path.exists(path):
            raise SystemExit(
                f"No trained model at {path}.\n"
                f"Train one first:  python train.py\n"
                f"Or play yourself: python play.py --human"
            )
        model = PPO.load(args.model_path)

    env = FlappyBirdEnv(render_mode="human")
    best = 0

    def pick_action(obs):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            return action
        return int(pygame.key.get_pressed()[pygame.K_SPACE])

    try:
        obs, _ = env.reset()
        screen = env.ensure_surface()  # opens the window for the menus to draw on
        clock = env.clock or pygame.time.Clock()

        # --- main menu -------------------------------------------------------
        if args.episodes is None:
            def draw_menu():
                env.draw_scene()
                _blit_centred(screen, message_image, HEIGHT // 2 - 40)

            if not _wait_for_key(env, draw_menu, clock):
                return

        episode = 0
        while True:
            episode += 1
            obs, _ = env.reset()
            done = False
            info = {"score": 0, "steps": 0}

            while not done:
                if not _pump():
                    return
                obs, _, terminated, truncated, info = env.step(pick_action(obs))
                done = terminated or truncated

            best = max(best, info["score"])
            print(f"episode {episode}: score={info['score']} steps={info['steps']}")

            if args.episodes is not None:
                if episode >= args.episodes:
                    return
                continue

            # --- game over ---------------------------------------------------
            def draw_gameover():
                env.draw_scene()
                _blit_centred(screen, gameover_image, HEIGHT // 2 - 70)
                # Final score full size, best-so-far smaller underneath it.
                draw_score(screen, info["score"], top=HEIGHT // 2 - 25)
                draw_score(screen, best, top=HEIGHT // 2 + 25, scale=0.6)

            if not _wait_for_key(env, draw_gameover, clock):
                return
    except SystemExit:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
