"""Level solvability checker using BFS."""

from collections import deque
from typing import Set, List, Optional, Tuple
import numpy as np
from core.game_state import GameState, Block


def is_solvable(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """
    Check if a level is solvable using BFS.
    Returns True if there's a valid path from start to goal.
    """
    try:
        initial_state = GameState.from_level(grid, start, goal)
    except (AssertionError, IndexError):
        return False
    
    if initial_state.block.pos1 == goal:
        return True
    
    visited: Set[Block] = set()
    queue = deque([initial_state])
    visited.add(initial_state.block)
    
    max_iterations = grid.size * 10  # Prevent infinite loops
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        current_state = queue.popleft()
        
        for action in current_state.valid_actions():
            next_state, _, done, _ = current_state.step(action)
            
            if done and next_state.success:
                return True
            
            if not done and next_state.block not in visited:
                visited.add(next_state.block)
                queue.append(next_state)
    
    return False


def find_solution(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[int]]:
    """
    Find a solution path from start to goal.
    Returns list of actions if solvable, None otherwise.
    """
    try:
        initial_state = GameState.from_level(grid, start, goal)
    except (AssertionError, IndexError):
        return None
    
    if initial_state.block.pos1 == goal:
        return []
    
    visited: Set[Block] = set()
    queue = deque([(initial_state, [])])
    visited.add(initial_state.block)
    
    max_iterations = grid.size * 10
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        current_state, path = queue.popleft()
        
        for action in current_state.valid_actions():
            next_state, _, done, _ = current_state.step(action)
            new_path = path + [action]
            
            if done and next_state.success:
                return new_path
            
            if not done and next_state.block not in visited:
                visited.add(next_state.block)
                queue.append((next_state, new_path))
    
    return None


def get_solution_length(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    Get the minimum number of steps to solve the level.
    Returns -1 if unsolvable.
    """
    solution = find_solution(grid, start, goal)
    return len(solution) if solution is not None else -1


def validate_level(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """
    Comprehensive level validation.
    Checks if level is well-formed and solvable.
    """
    h, w = grid.shape
    
    # Basic bounds checking
    if not (0 <= start[0] < w and 0 <= start[1] < h):
        return False
    if not (0 <= goal[0] < w and 0 <= goal[1] < h):
        return False
    
    # Start and goal must be on floor
    if not grid[start[1], start[0]] or not grid[goal[1], goal[0]]:
        return False
    
    # Start and goal must be different
    if start == goal:
        return False
    
    # Must have enough floor tiles for a meaningful puzzle
    floor_count = np.sum(grid)
    if floor_count < 3:
        return False
    
    # Check solvability
    return is_solvable(grid, start, goal)