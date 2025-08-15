"""Test script for trained Bloxorz DQN agent.
Supports both 2D and 3D visualization modes.
"""
import os
import argparse
import time
from typing import Optional

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from env.bloxorz_env import BloxorzEnv
from utils.helpers import set_global_seeds


def test_agent_2d(model_path: str, difficulty: str = "medium", episodes: int = 5, 
                  render: bool = True, seed: Optional[int] = None, speed: float = 0.1):
    """Test agent in 2D environment (same as training)."""
    print(f"Loading model from {model_path}...")
    
    # Create environment
    env = BloxorzEnv(
        difficulty=difficulty,
        render_mode="human" if render else None,
        seed=seed
    )
    
    # Wrap environment for model compatibility
    vec_env = DummyVecEnv([lambda: env])
    
    # Load trained model
    model = DQN.load(model_path, env=vec_env)
    
    total_reward = 0
    success_count = 0
    
    try:
        for episode in range(episodes):
            obs = vec_env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            
            while not done:
                # Predict action
                action, _states = model.predict(obs, deterministic=True)

                # Take step
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                step_count += 1

                if render:
                    time.sleep(speed)  # Slow down for visualization

                # Check if episode is done
                if done[0]:
                    if info[0].get('success', False):
                        success_count += 1
                        print(f"Success! Steps: {step_count}, Reward: {episode_reward:.2f}")
                    else:
                        print(f"Failed. Steps: {step_count}, Reward: {episode_reward:.2f}")
                    break
            
            total_reward += episode_reward
    
    finally:
        vec_env.close()
        avg_reward = total_reward / episodes if episodes > 0 else 0
        success_rate = success_count / episodes if episodes > 0 else 0

        print(f"\nResults:")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Success rate: {success_rate:.2%}")
    return avg_reward, success_rate




def main():
    parser = argparse.ArgumentParser(description="Test trained DQN agent on Bloxorz")
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"], 
                       help="Level difficulty")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--speed", type=float, default=0.1, help="Speed of the game (lower is faster)")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path + ".zip") and not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Set seed if provided
    if args.seed is not None:
        set_global_seeds(args.seed)
    
    # Run the test
    test_agent_2d(
        model_path=args.model_path,
        difficulty=args.difficulty,
        episodes=args.episodes,
        render=not args.no_render,
        seed=args.seed,
        speed=args.speed
    )


if __name__ == "__main__":
    main()