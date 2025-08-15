import argparse
import sys
import time

import pygame
from env.bloxorz_env import BloxorzEnv
from utils.config import FPS, BLOCK_SIZE, MAX_LEVEL_SIZE
from utils.helpers import set_global_seeds

def main():
    """Main function to run the human-playable game."""
    parser = argparse.ArgumentParser(description="Play Bloxorz manually")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"], help="Level difficulty")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for level generation")
    args = parser.parse_args()

    if args.seed is not None:
        set_global_seeds(args.seed)

    env = BloxorzEnv(difficulty=args.difficulty, render_mode="human", seed=args.seed)

    # Unused but needed for env.reset()
    obs, info = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action = -1
                if event.key == pygame.K_UP:
                    action = 0  # UP
                elif event.key == pygame.K_DOWN:
                    action = 1  # DOWN
                elif event.key == pygame.K_LEFT:
                    action = 2  # LEFT
                elif event.key == pygame.K_RIGHT:
                    action = 3  # RIGHT
                elif event.key == pygame.K_r:
                    print("Resetting level...")
                    env.reset()
                    continue
                elif event.key == pygame.K_q:
                    running = False
                    continue

                if action != -1:
                    obs, reward, terminated, truncated, info = env.step(action)

                    if terminated:
                        if info.get('success', False):
                            print("Success! You reached the goal.")
                        else:
                            print("Failure! You fell off the board.")

                        print("Press 'R' to play again or 'Q' to quit.")

                    if truncated:
                        print("Timeout! Too many steps.")
                        print("Press 'R' to play again or 'Q' to quit.")


    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
