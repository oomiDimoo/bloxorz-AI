from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import numpy as np

Action = int  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
ACTIONS: Dict[int, str] = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


@dataclass(frozen=True)
class Block:
    pos1: Tuple[int, int]
    pos2: Tuple[int, int]

    def __post_init__(self):
        # Ensure canonical ordering for hashability and equality
        p1 = self.pos1
        p2 = self.pos2
        if p2 < p1:
            object.__setattr__(self, "pos1", p2)
            object.__setattr__(self, "pos2", p1)

    @property
    def is_standing(self) -> bool:
        return self.pos1 == self.pos2

    @property
    def is_horizontal(self) -> bool:
        (x1, y1), (x2, y2) = self.pos1, self.pos2
        return y1 == y2 and abs(x2 - x1) == 1

    @property
    def is_vertical(self) -> bool:
        (x1, y1), (x2, y2) = self.pos1, self.pos2
        return x1 == x2 and abs(y2 - y1) == 1

    def occupied_cells(self) -> List[Tuple[int, int]]:
        if self.is_standing:
            return [self.pos1]
        return [self.pos1, self.pos2]

    def move(self, direction: str) -> Block:
        (x1, y1), (x2, y2) = self.pos1, self.pos2
        if self.is_standing:
            if direction == "LEFT":
                return Block((x1 - 2, y1), (x1 - 1, y1))
            if direction == "RIGHT":
                return Block((x1 + 1, y1), (x1 + 2, y1))
            if direction == "UP":
                return Block((x1, y1 - 2), (x1, y1 - 1))
            if direction == "DOWN":
                return Block((x1, y1 + 1), (x1, y1 + 2))
        elif self.is_horizontal:
            # pos1.x < pos2.x ensured by canonical order
            if direction == "LEFT":
                return Block((x1 - 1, y1), (x1 - 1, y1))  # stand on left outer
            if direction == "RIGHT":
                return Block((x2 + 1, y1), (x2 + 1, y1))  # stand on right outer
            if direction == "UP":
                return Block((x1, y1 - 1), (x2, y2 - 1))  # slide up
            if direction == "DOWN":
                return Block((x1, y1 + 1), (x2, y2 + 1))  # slide down
        elif self.is_vertical:
            # pos1.y < pos2.y ensured by canonical order
            if direction == "UP":
                return Block((x1, y1 - 1), (x1, y1 - 1))  # stand on top outer
            if direction == "DOWN":
                return Block((x2, y2 + 1), (x2, y2 + 1))  # stand on bottom outer
            if direction == "LEFT":
                return Block((x1 - 1, y1), (x2 - 1, y2))  # slide left
            if direction == "RIGHT":
                return Block((x1 + 1, y1), (x2 + 1, y2))  # slide right
        raise ValueError(f"Invalid direction {direction}")

    def in_bounds_and_valid(self, grid: np.ndarray) -> bool:
        h, w = grid.shape
        for (x, y) in self.occupied_cells():
            if x < 0 or y < 0 or x >= w or y >= h:
                return False
            if not grid[y, x]:
                return False
        return True


@dataclass
class GameState:
    grid: np.ndarray  # bool array: True for floor, False for void
    start: Tuple[int, int]
    goal: Tuple[int, int]
    block: Block
    steps: int = 0
    done: bool = False
    success: bool = False

    @staticmethod
    def from_level(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> "GameState":
        assert grid.dtype == np.bool_, "grid must be boolean array"
        assert grid[start[1], start[0]], "start must be on floor"
        assert grid[goal[1], goal[0]], "goal must be on floor"
        return GameState(grid=grid.copy(), start=start, goal=goal, block=Block(start, start))

    def copy(self) -> "GameState":
        return GameState(
            grid=self.grid.copy(),
            start=self.start,
            goal=self.goal,
            block=self.block,
            steps=self.steps,
            done=self.done,
            success=self.success,
        )

    def valid_actions(self) -> List[int]:
        acts: List[int] = []
        for a, name in ACTIONS.items():
            nb = self.block.move(name)
            if nb.in_bounds_and_valid(self.grid):
                acts.append(a)
        return acts

    def step(self, action: Action) -> Tuple["GameState", float, bool, Dict]:
        if self.done:
            return self, 0.0, True, {"terminated": True}
        direction = ACTIONS[action]
        next_block = self.block.move(direction)
        info: Dict = {"action_number": action, "action": direction}
        if not next_block.in_bounds_and_valid(self.grid):
            # fell off
            ns = self.copy()
            ns.block = next_block
            ns.steps += 1
            ns.done = True
            ns.success = False
            return ns, -1.0, True, info
        ns = self.copy()
        ns.block = next_block
        ns.steps += 1
        if ns.block.is_standing and ns.block.pos1 == self.goal:
            ns.done = True
            ns.success = True
            return ns, 1.0, True, info
        # small step penalty to encourage shorter solutions
        return ns, -0.01, False, info

    def is_terminal(self) -> bool:
        return self.done

    def encode_observation(self) -> np.ndarray:
        """
        Return a compact 3-channel observation (C,H,W):
        - channel 0: floor mask (1 where floor, 0 otherwise)
        - channel 1: block occupancy (1 where block occupies, 0 otherwise)
        - channel 2: goal mask (1 at goal, 0 otherwise)
        dtype: uint8
        """
        h, w = self.grid.shape
        obs = np.zeros((3, h, w), dtype=np.uint8)
        obs[0] = self.grid.astype(np.uint8)
        for (x, y) in self.block.occupied_cells():
            if 0 <= x < w and 0 <= y < h:
                obs[1, y, x] = 1
        gx, gy = self.goal
        obs[2, gy, gx] = 1
        return obs

    def manhattan_to_goal(self) -> int:
        x, y = self.block.pos1 if self.block.is_standing else self.block.pos2
        gx, gy = self.goal
        return abs(x - gx) + abs(y - gy)


def neighbors(state: GameState) -> List[Tuple[GameState, int]]:
    out: List[Tuple[GameState, int]] = []
    for a in state.valid_actions():
        ns, _, _, _ = state.step(a)
        if ns.block.in_bounds_and_valid(state.grid) and not ns.done:
            out.append((ns, a))
    return out