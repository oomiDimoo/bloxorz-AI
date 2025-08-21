"""
Optimized for GPU training with comprehensive monitoring.
"""
import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env.bloxorz_env import BloxorzEnv
from utils.helpers import set_global_seeds, create_directories
from utils.model import CustomCNN


class LatestModelCallback(BaseCallback):
    """Callback to save the latest model at a regular frequency."""
    def __init__(self, save_path: str, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, "latest_model.zip")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving latest model to {path}")
        return True


def create_training_env(config: dict, monitor_dir: str) -> DummyVecEnv:
    """Create and configure training environment."""
    def make_env():
        env = BloxorzEnv(
            difficulty=config["difficulty"],
            render_mode=None,  # No rendering during training
            seed=config["env_seed"]
        )
        env = Monitor(env, monitor_dir)
        return env
    
    # Create vectorized environment
    vec_env = DummyVecEnv([make_env])
    
    return vec_env


def create_eval_env(config: dict) -> DummyVecEnv:
    """Create evaluation environment."""
    def make_env():
        env = BloxorzEnv(
            difficulty=config["difficulty"],
            render_mode=None,
            seed=config["env_seed"] + 1000  # Different seed for evaluation
        )
        return env
    
    vec_env = DummyVecEnv([make_env])
    
    return vec_env


def create_model(env, config: dict, tensorboard_log: Optional[str] = None) -> DQN:
    """Create DQN model with optimized settings."""
    
    # Policy kwargs for network architecture
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": dict(features_dim=256),
        "net_arch": config["net_arch"],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
    }
    
    # Create model with GPU optimization
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        tau=1.0,  # Hard update
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_final_eps=config["exploration_final_eps"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        device=config["device"],
        verbose=1,
        seed=config["env_seed"]
    )
    
    return model


def setup_callbacks(config: dict, eval_env, save_path: str) -> CallbackList:
    """Setup training callbacks."""
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best_model"),
        log_path=os.path.join(save_path, "eval"),
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    if config["checkpoint_freq"] > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=config["checkpoint_freq"],
            save_path=os.path.join(save_path, "checkpoints"),
            name_prefix="bloxorz_dqn",
        )
        callbacks.append(checkpoint_callback)

    # Latest model callback
    if config["checkpoint_freq"] > 0:
        latest_model_callback = LatestModelCallback(
            save_path=os.path.join(save_path, "latest_model"),
            save_freq=config["checkpoint_freq"],
        )
        callbacks.append(latest_model_callback)

    return CallbackList(callbacks)


def optimize_gpu_settings():
    """Optimize GPU settings for training."""
    if torch.cuda.is_available():
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Set memory allocation strategy
        torch.cuda.empty_cache()

        # Print GPU info
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")

        return True
    else:
        print("CUDA not available. Training will use CPU.")
        return False


def main():
    # Load configuration from JSON file
    config_file_path = os.path.join(os.path.dirname(__file__), "training_config.json")
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    # Handle device selection
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Optimize GPU settings
    gpu_available = optimize_gpu_settings()
    if config["device"] == "cuda" and not gpu_available:
        print("Warning: CUDA was selected but is not available. Falling back to CPU.")
        config["device"] = "cpu"

    # Set global seeds for reproducibility
    set_global_seeds(config["env_seed"])

    # Create directories for saving models and logs
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(config["output_dir"], f"bloxorz_dqn_{current_time}")
    tensorboard_log = save_path if config["tensorboard_log"] else None
    create_directories([save_path, os.path.join(save_path, "eval"), os.path.join(save_path, "checkpoints"), os.path.join(save_path, "latest_model")])

    # Create environments
    train_env = create_training_env(config, monitor_dir=os.path.join(save_path, "monitor"))
    eval_env = create_eval_env(config)

    # Create model
    model = create_model(train_env, config, tensorboard_log)

    # Resume training if path is provided
    if config["resume_model_path"]:
        print(f"Loading model from {config["resume_model_path"]} for resume training...")
        model = DQN.load(config["resume_model_path"], env=train_env, device=config["device"])

    # Setup callbacks
    callbacks = setup_callbacks(config, eval_env, save_path)

    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            log_interval=config["log_interval"],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Save the final model
        final_model_path = os.path.join(save_path, "final_model.zip")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        train_env.close()
        eval_env.close()

    print("Training complete.")

if __name__ == "__main__":
    main()