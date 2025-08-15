"""Main training script for Bloxorz DQN agent.
Optimized for GPU training with comprehensive monitoring.
"""
import os
import argparse
import time
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env.bloxorz_env import BloxorzEnv
from utils.helpers import set_global_seeds, create_directories
from utils.callbacks import (
    TensorboardCallback, ProgressCallback, CustomEvalCallback
)


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Environment settings
        self.difficulty = "medium"
        self.env_seed = 42
        
        # Training settings
        self.total_timesteps = 1_000_000
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.buffer_size = 100_000
        self.learning_starts = 10_000
        self.target_update_interval = 1000
        self.train_freq = 4
        self.gradient_steps = 1
        self.exploration_fraction = 0.1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.05
        self.gamma = 0.99
        
        # Network architecture
        self.net_arch = [256, 256]
        
        # GPU settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_mixed_precision = True
        
        # Evaluation settings
        self.eval_freq = 10_000
        self.eval_episodes = 10
        
        # Checkpoint settings
        self.checkpoint_freq = 50_000
        self.save_best_model = True
        
        # Logging
        self.log_interval = 1000
        self.tensorboard_log = True
        
    def update_from_args(self, args):
        """Update configuration from command line arguments."""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
    
    def __str__(self):
        """String representation of configuration."""
        config_str = "Training Configuration:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str


def create_training_env(config: TrainingConfig, monitor_dir: str) -> DummyVecEnv:
    """Create and configure training environment."""
    def make_env():
        env = BloxorzEnv(
            difficulty=config.difficulty,
            render_mode=None,  # No rendering during training
            seed=config.env_seed
        )
        env = Monitor(env, monitor_dir)
        return env
    
    # Create vectorized environment
    vec_env = DummyVecEnv([make_env])
    vec_env = VecTransposeImage(vec_env)
    
    return vec_env


def create_eval_env(config: TrainingConfig) -> DummyVecEnv:
    """Create evaluation environment."""
    def make_env():
        env = BloxorzEnv(
            difficulty=config.difficulty,
            render_mode=None,
            seed=config.env_seed + 1000  # Different seed for evaluation
        )
        return env
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecTransposeImage(vec_env)
    
    return vec_env


def create_model(env, config: TrainingConfig, tensorboard_log: Optional[str] = None) -> DQN:
    """Create DQN model with optimized settings."""
    
    # Policy kwargs for network architecture
    policy_kwargs = {
        "net_arch": config.net_arch,
        "activation_fn": torch.nn.ReLU,
        "normalize_images": True,
    }
    
    # Create model with GPU optimization
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=1.0,  # Hard update
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        device=config.device,
        verbose=1,
        seed=config.env_seed
    )
    
    return model


def setup_callbacks(config: TrainingConfig, eval_env, save_path: str, 
                   tensorboard_log: str) -> CallbackList:
    """Setup training callbacks."""
    callbacks = []
    
    # Progress callback
    callbacks.append(ProgressCallback(
        log_interval=config.log_interval,
        save_path=save_path
    ))
    
    # Evaluation callback
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "best_model"),
        log_path=os.path.join(save_path, "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix="bloxorz_dqn"
    )
    callbacks.append(checkpoint_callback)
    
    # Tensorboard callback
    if config.tensorboard_log:
        callbacks.append(TensorboardCallback(
            log_dir=tensorboard_log,
            model_save_path=save_path
        ))
    
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
    parser = argparse.ArgumentParser(description="Train DQN agent on Bloxorz")
    
    # Environment arguments
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Level difficulty")
    parser.add_argument("--env-seed", type=int, default=42, help="Environment seed")
    
    # Training arguments
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, 
                       help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100_000, 
                       help="Replay buffer size")
    
    # Model arguments
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256], 
                       help="Network architecture")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--no-mixed-precision", action="store_true", 
                       help="Disable mixed precision training")
    
    # Evaluation arguments
    parser.add_argument("--eval-freq", type=int, default=10_000, 
                       help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=10, 
                       help="Number of evaluation episodes")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, 
                       help="Checkpoint save frequency")
    
    # Logging arguments
    parser.add_argument("--log-interval", type=int, default=1000, 
                       help="Logging interval")
    parser.add_argument("--no-tensorboard", action="store_true", 
                       help="Disable tensorboard logging")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to model to resume training")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="./training_output", 
                       help="Output directory for models and logs")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig()
    config.update_from_args(args)
    
    # Handle device selection
    if args.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        config.device = args.device
    
    # Handle mixed precision
    config.use_mixed_precision = not args.no_mixed_precision
    
    # Handle tensorboard
    config.tensorboard_log = not args.no_tensorboard
    
    print(config)
    
    # Optimize GPU settings
    gpu_available = optimize_gpu_settings()
    if config.device == "cuda" and not gpu_available:
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        config.device = "cpu"
    
    # Set global seeds
    set_global_seeds(config.env_seed)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, f"bloxorz_dqn_{timestamp}")
    tensorboard_log = os.path.join(save_path, "tensorboard") if config.tensorboard_log else None
    monitor_dir = os.path.join(save_path, "monitor")
    
    create_directories([save_path, monitor_dir])
    if tensorboard_log:
        create_directories([tensorboard_log])
    
    print(f"\nSaving results to: {save_path}")
    
    # Save configuration
    config_path = os.path.join(save_path, "config.txt")
    with open(config_path, "w") as f:
        f.write(str(config))
    
    # Create environments
    print("\nCreating environments...")
    train_env = create_training_env(config, monitor_dir)
    eval_env = create_eval_env(config)
    
    # Create or load model
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        model = DQN.load(args.resume, env=train_env, device=config.device)
        # Update some parameters for resumed training
        model.learning_rate = config.learning_rate
    else:
        print("\nCreating new model...")
        model = create_model(train_env, config, tensorboard_log)
    
    # Setup callbacks
    callbacks = setup_callbacks(config, eval_env, save_path, tensorboard_log)
    
    # Start training
    print("\nStarting training...")
    print(f"Device: {config.device}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Save final model
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    
    print(f"\nTraining completed!")
    print(f"Training time: {training_time / 3600:.2f} hours")
    print(f"Final model saved to: {final_model_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Difficulty: {config.difficulty}")
    print(f"  Total timesteps: {config.total_timesteps:,}")
    print(f"  Device: {config.device}")
    print(f"  Training time: {training_time / 3600:.2f} hours")
    print(f"  Models saved in: {save_path}")
    
    if tensorboard_log:
        print(f"\nTo view training progress:")
        print(f"  tensorboard --logdir {tensorboard_log}")


if __name__ == "__main__":
    main()