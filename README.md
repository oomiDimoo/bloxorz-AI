# Bloxorz AI Game

A Python implementation of the classic Bloxorz puzzle game with AI training capabilities using Deep Q-Network (DQN) from Stable Baselines 3.

## Features

- **2D Training Mode**: Optimized for fast AI training with minimal computational overhead
- **3D Gameplay Mode**: Full 3D OpenGL rendering for human players
- **Procedural Level Generation**: Dynamic level creation ensuring solvable puzzles
- **GPU-Optimized Training**: CUDA support for accelerated DQN training
- **Modular Architecture**: Clean separation of game logic, rendering, and AI components

## Project Structure

```
bloxorz_ai/
├── core/                   # Core game logic
│   ├── game_engine.py     # Main game engine
│   ├── block.py           # Block physics and movement
│   ├── level.py           # Level representation and validation
│   └── game_state.py      # Game state management
├── rendering/              # Rendering systems
│   ├── renderer_2d.py     # 2D pygame renderer for training
│   ├── renderer_3d.py     # 3D OpenGL renderer for gameplay
│   └── camera.py          # Camera controls for 3D view
├── ai/                     # AI training components
│   ├── environment.py     # Gym environment wrapper
│   ├── dqn_trainer.py     # DQN training script
│   └── rewards.py         # Reward function definitions
├── level_generation/       # Procedural level generation
│   ├── generator.py       # Main level generator
│   ├── solver.py          # Level solvability checker
│   └── patterns.py        # Level pattern library
├── utils/                  # Utility functions
│   ├── config.py          # Game configuration
│   └── helpers.py         # Helper functions
├── main.py                 # Main game launcher
├── train_ai.py            # AI training script
└── test_game.py           # Game testing script
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the game:
```bash
python main.py
```

3. Train AI:
```bash
python train_ai.py
```

## Controls

- **Arrow Keys**: Move the block
- **R**: Reset level
- **N**: Generate new level
- **ESC**: Exit game

## Training

The AI training uses DQN with the following optimizations:
- GPU acceleration when available
- Vectorized environments for parallel training
- Custom reward shaping for faster convergence
- Curriculum learning with progressively harder levels