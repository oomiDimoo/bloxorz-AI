"""Training script for Bloxorz DQN agent using stable-baselines3.
Optimized for GPU training with proper environment wrappers.
"""
import os
import argparse
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env.bloxorz_env import BloxorzEnv
from utils.helpers import set_global_seeds
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.action_number = 0

    def _on_step(self) -> bool:
        if "action_number" in self.locals["infos"][0]:
            self.action_number = self.locals["infos"][0]["action_number"]
        return True


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def create_env(difficulty: str = "medium", max_steps: Optional[int] = None, seed: Optional[int] = None):
    """Create a single Bloxorz environment."""
    def _init():
        env = BloxorzEnv(
            difficulty=difficulty,
            max_steps=max_steps,
            render_mode=None,  # No rendering during training
            seed=seed
        )
        env = Monitor(env)  # For logging
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent on Bloxorz")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"], help="Level difficulty")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=50000, help="Model save frequency")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--target-update-interval", type=int, default=1000, help="Target network update interval")
    parser.add_argument("--exploration-fraction", type=float, default=0.3, help="Exploration fraction")
    parser.add_argument("--exploration-final-eps", type=float, default=0.05, help="Final exploration epsilon")
    
    args = parser.parse_args()
    
    # Set random seeds
    set_global_seeds(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create training environments
    train_env = make_vec_env(
        create_env(difficulty=args.difficulty, seed=args.seed),
        n_envs=args.n_envs,
        seed=args.seed
    )
    
    # Create evaluation environment
    eval_env = make_vec_env(
        create_env(difficulty=args.difficulty, seed=args.seed + 1000),
        n_envs=1,
        seed=args.seed + 1000
    )

    
    # DQN configuration optimized for GPU
    model_config = {
        "policy": "CnnPolicy",
        "env": train_env,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": 1000,
        "batch_size": args.batch_size,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": args.exploration_final_eps,
        "max_grad_norm": 10,
        "tensorboard_log": args.log_dir,
        "device": device,
        "verbose": 1,
        "seed": args.seed,
    }
    
    # CNN policy configuration for better GPU utilization
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": 512,
        },
        "net_arch": [512, 512],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,  # Already normalized in env
    }
    model_config["policy_kwargs"] = policy_kwargs
    
    # Create DQN model
    model = DQN(**model_config)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best_model"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=os.path.join(args.save_dir, "checkpoints"),
        name_prefix="dqn_bloxorz",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    callbacks = [eval_callback, checkpoint_callback]

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=100,
            tb_log_name="DQN_Bloxorz",
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main()