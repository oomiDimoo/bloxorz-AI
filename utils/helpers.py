"""Helper utilities for seeding, observation processing, and misc."""
import numpy as np
import random
import torch


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    # Convert uint8 [0,1] to float32 [0,1]
    if obs.dtype != np.float32:
        return obs.astype(np.float32)
    return obs


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])