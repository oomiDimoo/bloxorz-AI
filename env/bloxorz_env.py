"""Gymnasium environment for Bloxorz clone (2D training-friendly).
- Fixed-size observations by padding levels to MAX_LEVEL_SIZE.
- Render modes: "rgb_array" (numpy) and optional "human" via pygame.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from core.game_state import GameState, ACTIONS
from level_generation.generator import LevelGenerator
from utils.config import MAX_LEVEL_SIZE, FPS, BLOCK_SIZE, COLORS
from utils.helpers import set_global_seeds

try:
    import pygame  # Optional for human render
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False


class BloxorzEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": FPS}

    def __init__(self, difficulty: str = "medium", max_steps: Optional[int] = None, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.generator = LevelGenerator(seed=seed)
        self.rng = np.random.RandomState(seed if seed is not None else 0)
        if seed is not None:
            set_global_seeds(seed)
        self.max_steps = max_steps if max_steps is not None else 5 * MAX_LEVEL_SIZE * MAX_LEVEL_SIZE

        # Fixed observation shape: (3, H, W)
        self.obs_shape = (3, MAX_LEVEL_SIZE, MAX_LEVEL_SIZE)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

        # Runtime state
        self.state: Optional[GameState] = None
        self._steps = 0

        # For human rendering
        self._screen = None
        self._clock = None
        self._sprites: Dict[str, pygame.Surface] = {}

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            set_global_seeds(seed)
            self.generator = LevelGenerator(seed=seed)
            self.rng = np.random.RandomState(seed)

    def _pad_level(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        h, w = grid.shape
        H, W = MAX_LEVEL_SIZE, MAX_LEVEL_SIZE
        padded = np.zeros((H, W), dtype=bool)
        # place level at top-left corner
        padded[:h, :w] = grid
        sx, sy = start
        gx, gy = goal
        return padded, (sx, sy), (gx, gy)

    def _get_obs(self) -> np.ndarray:
        obs = self.state.encode_observation()  # (3,h,w) uint8
        # pad to fixed size if needed
        H, W = MAX_LEVEL_SIZE, MAX_LEVEL_SIZE
        _, h, w = obs.shape
        if h == H and w == W:
            return obs
        padded = np.zeros((3, H, W), dtype=np.uint8)
        padded[:, :h, :w] = obs
        return padded

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        diff = options.get("difficulty") if options else None
        difficulty = diff if isinstance(diff, str) else self.difficulty

        grid, start, goal = self.generator.generate_level(difficulty)
        grid, start, goal = self._pad_level(grid, start, goal)
        self.state = GameState.from_level(grid, start, goal)
        self._steps = 0

        if self.render_mode == "human":
            self._ensure_pygame()
            self._draw_human()
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()"
        self._steps += 1
        next_state, reward, done, info = self.state.step(int(action))
        self.state = next_state

        terminated = done
        truncated = self._steps >= self.max_steps and not terminated
        if truncated:
            done = True

        if self.render_mode == "human":
            self._draw_human()
        elif self.render_mode == "rgb_array":
            self.render()
        return self._get_obs(), reward, terminated, truncated, {**info, "success": self.state.success}

    # ---------- Rendering ----------
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            self._draw_human()
            return None
        return None

    def _render_rgb_array(self) -> np.ndarray:
        assert self.state is not None
        H, W = MAX_LEVEL_SIZE, MAX_LEVEL_SIZE
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # base colors
        bg = np.array(COLORS.get("background", (50, 50, 50)), dtype=np.uint8)
        floor = np.array(COLORS.get("floor", (200, 200, 200)), dtype=np.uint8)
        blockc = np.array(COLORS.get("block", (255, 100, 100)), dtype=np.uint8)
        goalc = np.array(COLORS.get("target", (100, 255, 100)), dtype=np.uint8)
        img[:, :] = bg
        grid = self.state.grid
        h, w = grid.shape
        # draw floor
        ys, xs = np.where(grid)
        img[ys, xs] = floor
        # draw goal
        gx, gy = self.state.goal
        img[gy, gx] = goalc
        # draw block
        for (x, y) in self.state.block.occupied_cells():
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = blockc
        # scale to pixels
        k = max(1, BLOCK_SIZE // 4)
        return np.kron(img, np.ones((k, k, 1), dtype=np.uint8))

    def _ensure_pygame(self):
        if not _HAS_PYGAME:
            return
        if self._screen is None:
            pygame.init()
            H, W = MAX_LEVEL_SIZE, MAX_LEVEL_SIZE
            self._screen = pygame.display.set_mode((W * BLOCK_SIZE, H * BLOCK_SIZE))
            pygame.display.set_caption("Bloxorz 2D")
            self._clock = pygame.time.Clock()

            # Load sprites
            if self.render_mode == "human":
                try:
                    self._sprites["floor"] = pygame.image.load("assets/images/floor.png").convert_alpha()
                    self._sprites["block"] = pygame.image.load("assets/images/block_face.png").convert_alpha()
                    self._sprites["goal"] = pygame.image.load("assets/images/goal.png").convert_alpha()

                    # Scale sprites
                    for key, sprite in self._sprites.items():
                        self._sprites[key] = pygame.transform.scale(sprite, (BLOCK_SIZE, BLOCK_SIZE))
                except pygame.error as e:
                    print(f"Warning: Could not load sprites. Using default rendering. Error: {e}")
                    self._sprites = {}  # Fallback to default rendering

    def _draw_human(self):
        if not _HAS_PYGAME or self._screen is None:
            return

        # If sprites are not loaded, fallback to rgb_array rendering
        if not self._sprites:
            surf = pygame.surfarray.make_surface(self._render_rgb_array().swapaxes(0, 1))
            self._screen.blit(surf, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.metadata.get("render_fps", 60))
            return

        H, W = MAX_LEVEL_SIZE, MAX_LEVEL_SIZE

        # Background
        self._screen.fill(COLORS.get("background", (50, 50, 50)))

        # Draw floor
        assert self.state is not None
        grid = self.state.grid
        for r in range(H):
            for c in range(W):
                if grid[r, c]:
                    self._screen.blit(self._sprites["floor"], (c * BLOCK_SIZE, r * BLOCK_SIZE))

        # Draw goal
        gx, gy = self.state.goal
        self._screen.blit(self._sprites["goal"], (gx * BLOCK_SIZE, gy * BLOCK_SIZE))

        # Draw block
        for (x, y) in self.state.block.occupied_cells():
            if 0 <= x < W and 0 <= y < H:
                self._screen.blit(self._sprites["block"], (x * BLOCK_SIZE, y * BLOCK_SIZE))

        pygame.display.flip()
        self._clock.tick(self.metadata.get("render_fps", 60))

    def close(self):
        if _HAS_PYGAME and self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None