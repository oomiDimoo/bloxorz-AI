"""Game configuration settings."""

import torch

# Game settings
GRID_SIZE = 20
BLOCK_SIZE = 32  # Size in pixels for 2D rendering
FPS = 60

# Colors (RGB)
COLORS = {
    'background': (50, 50, 50),
    'grid': (100, 100, 100),
    'block': (255, 100, 100),
    'target': (100, 255, 100),
    'hole': (0, 0, 0),
    'floor': (200, 200, 200),
    'wall': (150, 150, 150)
}

# 3D rendering settings
CAMERA_DISTANCE = 30.0
CAMERA_HEIGHT = 20.0
FOV = 45.0

# AI training settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Level generation settings
MIN_LEVEL_SIZE = 5
MAX_LEVEL_SIZE = 15
MIN_PATH_LENGTH = 8
MAX_PATH_LENGTH = 30
MAX_GENERATION_ATTEMPTS = 100

# Movement directions
DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}

# Block orientations
ORIENTATIONS = {
    'STANDING': 'standing',
    'LYING_HORIZONTAL': 'lying_horizontal',
    'LYING_VERTICAL': 'lying_vertical'
}