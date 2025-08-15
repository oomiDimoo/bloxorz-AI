"""Procedural level generator with solvability guarantees."""
import random
from typing import Tuple, Optional
import numpy as np
from .patterns import random_walk, add_noise_tiles
from .solver import validate_level, get_solution_length
from utils.config import MIN_LEVEL_SIZE, MAX_LEVEL_SIZE, MIN_PATH_LENGTH, MAX_PATH_LENGTH, MAX_GENERATION_ATTEMPTS


class LevelGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_level(self, difficulty: str = "medium") -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Generate a solvable level with specified difficulty.
        Returns (grid, start_pos, goal_pos)
        """
        difficulty_params = self._get_difficulty_params(difficulty)
        
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                level_data = self._generate_single_attempt(difficulty_params)
                if level_data is not None:
                    return level_data
            except Exception:
                continue
        
        # Fallback: generate a simple linear level
        return self._generate_fallback_level()
    
    def _get_difficulty_params(self, difficulty: str) -> dict:
        """Get generation parameters based on difficulty."""
        if difficulty == "easy":
            return {
                "size_range": (5, 8),
                "path_length_range": (8, 15),
                "noise_ratio": 0.1,
                "min_solution_length": 6
            }
        elif difficulty == "medium":
            return {
                "size_range": (8, 12),
                "path_length_range": (15, 25),
                "noise_ratio": 0.25,
                "min_solution_length": 10
            }
        elif difficulty == "hard":
            return {
                "size_range": (10, 15),
                "path_length_range": (20, 30),
                "noise_ratio": 0.4,
                "min_solution_length": 15
            }
        else:
            return self._get_difficulty_params("medium")
    
    def _generate_single_attempt(self, params: dict) -> Optional[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
        """Single attempt at level generation."""
        # Random dimensions
        w = random.randint(*params["size_range"])
        h = random.randint(*params["size_range"])
        
        # Generate main path using random walk
        start = (random.randint(1, w-2), random.randint(1, h-2))
        path = random_walk(start, w, h, *params["path_length_range"])
        
        if len(path) < params["min_solution_length"]:
            return None
        
        # Add noise tiles for complexity
        all_tiles = add_noise_tiles(path, w, h, params["noise_ratio"])
        
        # Create grid
        grid = np.zeros((h, w), dtype=bool)
        for x, y in all_tiles:
            grid[y, x] = True
        
        # Select start and goal
        start_pos = path[0]
        goal_pos = path[-1]
        
        # Ensure minimum distance between start and goal
        if abs(start_pos[0] - goal_pos[0]) + abs(start_pos[1] - goal_pos[1]) < 3:
            return None
        
        # Validate level
        if not validate_level(grid, start_pos, goal_pos):
            return None
        
        # Check solution length meets minimum requirement
        solution_length = get_solution_length(grid, start_pos, goal_pos)
        if solution_length < params["min_solution_length"]:
            return None
        
        return grid, start_pos, goal_pos
    
    def _generate_fallback_level(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Generate a simple guaranteed-solvable level as fallback."""
        # Create a simple 7x5 linear path
        grid = np.zeros((5, 7), dtype=bool)
        
        # Create L-shaped path
        for x in range(5):
            grid[2, x] = True  # horizontal line
        for y in range(3):
            grid[y, 4] = True  # vertical line
        
        start_pos = (0, 2)
        goal_pos = (4, 0)
        
        return grid, start_pos, goal_pos
    
    def generate_curriculum_level(self, progress: float) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Generate level for curriculum learning.
        progress: 0.0 (easiest) to 1.0 (hardest)
        """
        if progress < 0.3:
            difficulty = "easy"
        elif progress < 0.7:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        return self.generate_level(difficulty)


# Convenience function
def generate_random_level(difficulty: str = "medium", seed: Optional[int] = None) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Generate a single random level."""
    generator = LevelGenerator(seed)
    return generator.generate_level(difficulty)