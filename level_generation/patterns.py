"""Reusable level pattern utilities for procedural generation."""
from typing import List, Tuple, Set
import random

Coord = Tuple[int, int]


def random_walk(start: Coord, w: int, h: int, min_len: int, max_len: int, avoid_backtrack: bool = True) -> List[Coord]:
    """Generate a simple self-avoiding random walk inside [0,w) x [0,h)."""
    path: List[Coord] = [start]
    visited: Set[Coord] = {start}
    target_len = random.randint(min_len, max_len)

    while len(path) < target_len:
        x, y = path[-1]
        nbrs = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        random.shuffle(nbrs)
        moved = False
        for nx, ny in nbrs:
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                if avoid_backtrack and len(path) > 1 and (nx, ny) == path[-2]:
                    continue
                path.append((nx, ny))
                visited.add((nx, ny))
                moved = True
                break
        if not moved:
            # dead end; backtrack if possible
            if len(path) > 1:
                path.pop()
            else:
                break
    return path


def add_noise_tiles(path: List[Coord], w: int, h: int, noise_ratio: float = 0.2, max_attempts: int = 200) -> Set[Coord]:
    """Add random adjacent floor tiles branching off from path to create variety."""
    tiles: Set[Coord] = set(path)
    attempts = 0
    target_extra = int(len(path) * noise_ratio)
    while attempts < max_attempts and len(tiles) < len(path) + target_extra:
        attempts += 1
        x, y = random.choice(list(tiles))
        nbrs = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        nx, ny = random.choice(nbrs)
        if 0 <= nx < w and 0 <= ny < h:
            tiles.add((nx, ny))
    return tiles