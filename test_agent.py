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
from game.bloxorz_3d import Bloxorz3DGame
from utils.helpers import set_global_seeds


def test_agent_2d(model_path: str, difficulty: str = "medium", episodes: int = 5, 
                  render: bool = True, seed: Optional[int] = None):
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
    vec_env = VecTransposeImage(vec_env)
    
    # Load trained model
    model = DQN.load(model_path, env=vec_env)
    
    total_reward = 0
    success_count = 0
    
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
                time.sleep(0.1)  # Slow down for visualization
            
            # Check if episode is done
            if done[0]:
                if info[0].get('success', False):
                    success_count += 1
                    print(f"Success! Steps: {step_count}, Reward: {episode_reward:.2f}")
                else:
                    print(f"Failed. Steps: {step_count}, Reward: {episode_reward:.2f}")
                break
        
        total_reward += episode_reward
    
    avg_reward = total_reward / episodes
    success_rate = success_count / episodes
    
    print(f"\nResults:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Success rate: {success_rate:.2%}")
    
    vec_env.close()
    return avg_reward, success_rate


def test_agent_3d(model_path: str, difficulty: str = "medium", episodes: int = 5, 
                  seed: Optional[int] = None):
    """Test agent in 3D environment for better visualization."""
    print(f"Loading model from {model_path}...")
    
    # Create 2D environment for prediction (model was trained on this)
    train_env = BloxorzEnv(
        difficulty=difficulty,
        render_mode=None,
        seed=seed
    )
    vec_env = DummyVecEnv([lambda: train_env])
    vec_env = VecTransposeImage(vec_env)
    
    # Load trained model
    model = DQN.load(model_path, env=vec_env)
    
    total_reward = 0
    success_count = 0
    
    for episode in range(episodes):
        # Create 3D game for visualization
        game_3d = Bloxorz3DGame(difficulty=difficulty, seed=seed)
        
        # Reset 2D environment for predictions
        obs = vec_env.reset()
        
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        print("Close the 3D window to continue to next episode")
        
        while game_3d.running:
            # Predict action using 2D environment
            action, _states = model.predict(obs, deterministic=True)
            action_name = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action[0]]
            
            # Apply action to 3D game
            reward, done, success = game_3d.step(action[0])
            
            # Apply same action to 2D environment for next prediction
            obs, _, _, _ = vec_env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Handle game events
            game_3d.handle_events()
            game_3d.render()
            
            if done:
                if success:
                    success_count += 1
                    print(f"Success! Steps: {step_count}, Reward: {episode_reward:.2f}")
                else:
                    print(f"Failed. Steps: {step_count}, Reward: {episode_reward:.2f}")
                time.sleep(2)  # Show result for 2 seconds
                break
            
            time.sleep(0.3)  # Slow down for better visualization
        
        total_reward += episode_reward
        game_3d.quit()
    
    avg_reward = total_reward / episodes
    success_rate = success_count / episodes
    
    print(f"\nResults:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Success rate: {success_rate:.2%}")
    
    vec_env.close()
    return avg_reward, success_rate


def interactive_3d_test(model_path: str, difficulty: str = "medium", seed: Optional[int] = None):
    """Interactive 3D testing where user can choose between AI and manual control."""
    print(f"Loading model from {model_path}...")
    
    # Create 2D environment for AI predictions
    train_env = BloxorzEnv(
        difficulty=difficulty,
        render_mode=None,
        seed=seed
    )
    vec_env = DummyVecEnv([lambda: train_env])
    vec_env = VecTransposeImage(vec_env)
    
    # Load trained model
    model = DQN.load(model_path, env=vec_env)
    
    # Create 3D game
    game_3d = Bloxorz3DGame(difficulty=difficulty, seed=seed)
    
    obs = vec_env.reset()
    ai_mode = False
    
    print("\nInteractive 3D Test Mode")
    print("Controls:")
    print("  Arrow Keys: Manual control")
    print("  SPACE: Toggle AI mode")
    print("  R: Reset level")
    print("  ESC: Quit")
    print(f"\nAI Mode: {'ON' if ai_mode else 'OFF'}")
    
    while game_3d.running:
        action = None
        
        # Handle events
        for event in game_3d.handle_events():
            if event == 'toggle_ai':
                ai_mode = not ai_mode
                print(f"AI Mode: {'ON' if ai_mode else 'OFF'}")
            elif event == 'reset':
                game_3d.reset()
                obs = vec_env.reset()
                print("Level reset")
            elif event in ['UP', 'DOWN', 'LEFT', 'RIGHT'] and not ai_mode:
                action = ['UP', 'DOWN', 'LEFT', 'RIGHT'].index(event)
        
        # AI prediction
        if ai_mode and action is None:
            predicted_action, _states = model.predict(obs, deterministic=True)
            action = predicted_action[0]
            time.sleep(0.5)  # Slow down AI for better observation
        
        # Apply action if any
        if action is not None:
            reward, done, success = game_3d.step(action)
            obs, _, _, _ = vec_env.step([action])
            
            if done:
                if success:
                    print("Level completed!")
                else:
                    print("Game over!")
                time.sleep(1)
                game_3d.reset()
                obs = vec_env.reset()
        
        game_3d.render()
    
    game_3d.quit()
    vec_env.close()


def main():
    parser = argparse.ArgumentParser(description="Test trained DQN agent on Bloxorz")
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("--mode", type=str, default="2d", choices=["2d", "3d", "interactive"], 
                       help="Testing mode (2d/3d/interactive)")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"], 
                       help="Level difficulty")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering (2D mode only)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path + ".zip") and not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Set seed if provided
    if args.seed is not None:
        set_global_seeds(args.seed)
    
    # Run appropriate test mode
    if args.mode == "2d":
        test_agent_2d(
            model_path=args.model_path,
            difficulty=args.difficulty,
            episodes=args.episodes,
            render=not args.no_render,
            seed=args.seed
        )
    elif args.mode == "3d":
        test_agent_3d(
            model_path=args.model_path,
            difficulty=args.difficulty,
            episodes=args.episodes,
            seed=args.seed
        )
    elif args.mode == "interactive":
        interactive_3d_test(
            model_path=args.model_path,
            difficulty=args.difficulty,
            seed=args.seed
        )


if __name__ == "__main__":
    main()