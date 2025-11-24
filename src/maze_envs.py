"""
Maze Environment for Transformer vs RNN Comparison

This module provides maze generation, visualization, and pathfinding utilities
to demonstrate the difference between sequential (RNN) and parallel (Transformer)
processing approaches.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class MazeConfig:
    """Configuration for maze generation"""
    height: int = 15
    width: int = 15
    wall_probability: float = 0.3
    ensure_solvable: bool = True
    seed: Optional[int] = None


class Maze:
    """
    A maze environment that supports both numerical and text representations.
    
    Numerical representation:
        0 = wall
        1 = path
        2 = start
        3 = goal
        
    Text representation:
        # = wall
        . = path
        S = start
        G = goal
        * = solution path
    """
    
    def __init__(self, config: MazeConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        
        self.grid = self._generate_maze()
        self.start = self._place_special_cell(2)
        self.goal = self._place_special_cell(3)
        
        if config.ensure_solvable:
            self._ensure_path_exists()
        
        self.solution = None
        
    def _generate_maze(self) -> np.ndarray:
        """Generate a random maze grid"""
        h, w = self.config.height, self.config.width
        
        # Start with all paths
        grid = np.ones((h, w), dtype=int)
        
        # Add walls randomly (but keep borders as walls for cleaner visualization)
        grid[0, :] = 0
        grid[-1, :] = 0
        grid[:, 0] = 0
        grid[:, -1] = 0
        
        # Add random internal walls
        for i in range(1, h-1):
            for j in range(1, w-1):
                if np.random.random() < self.config.wall_probability:
                    grid[i, j] = 0
                    
        return grid
    
    def _place_special_cell(self, value: int) -> Tuple[int, int]:
        """Place start or goal in a random path cell"""
        path_cells = np.argwhere(self.grid == 1)
        if len(path_cells) == 0:
            raise ValueError("No path cells available for placement")
        
        idx = np.random.choice(len(path_cells))
        pos = tuple(path_cells[idx])
        self.grid[pos] = value
        return pos
    
    def _ensure_path_exists(self):
        """Ensure there's a valid path from start to goal using BFS"""
        if not self.is_solvable():
            # Carve a path using A* style approach
            self._carve_path()
            
    def is_solvable(self) -> bool:
        """Check if maze is solvable using BFS"""
        return self.solve() is not None
    
    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to goal using BFS.
        Returns list of (row, col) positions or None if unsolvable.
        """
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos == self.goal:
                self.solution = path
                return path
            
            for next_pos in self._get_neighbors(pos):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return None
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (up, down, left, right)"""
        row, col = pos
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < self.config.height and 
                0 <= new_col < self.config.width and
                self.grid[new_row, new_col] != 0):  # Not a wall
                neighbors.append((new_row, new_col))
                
        return neighbors
    
    def _carve_path(self):
        """Carve a guaranteed path from start to goal"""
        current = self.start
        
        while current != self.goal:
            row, col = current
            goal_row, goal_col = self.goal
            
            # Move toward goal
            if row < goal_row:
                next_pos = (row + 1, col)
            elif row > goal_row:
                next_pos = (row - 1, col)
            elif col < goal_col:
                next_pos = (row, col + 1)
            else:
                next_pos = (row, col - 1)
            
            # Carve path
            if self.grid[next_pos] == 0:
                self.grid[next_pos] = 1
            
            current = next_pos
    
    def to_text(self, show_solution: bool = False) -> str:
        """
        Convert maze to text representation.
        
        Args:
            show_solution: If True and solution exists, mark solution path with *
        """
        char_map = {
            0: '#',  # wall
            1: '.',  # path
            2: 'S',  # start
            3: 'G',  # goal
        }
        
        lines = []
        solution_set = set(self.solution) if (show_solution and self.solution) else set()
        
        for i in range(self.config.height):
            row = []
            for j in range(self.config.width):
                if (i, j) in solution_set and self.grid[i, j] == 1:
                    row.append('*')
                else:
                    row.append(char_map[self.grid[i, j]])
            lines.append(''.join(row))
        
        return '\n'.join(lines)
    
    def get_path_length(self) -> Optional[int]:
        """Get length of solution path (if it exists)"""
        if self.solution is None:
            self.solve()
        return len(self.solution) if self.solution else None
    
    def to_sequence(self) -> List[str]:
        """
        Convert maze to a sequence representation for language-model-style processing.
        Returns list of tokens representing the maze state.
        """
        tokens = []
        for i in range(self.config.height):
            for j in range(self.config.width):
                cell = self.grid[i, j]
                if cell == 0:
                    tokens.append('WALL')
                elif cell == 1:
                    tokens.append('PATH')
                elif cell == 2:
                    tokens.append('START')
                elif cell == 3:
                    tokens.append('GOAL')
        return tokens
    
    def actions_to_path(self, start: Tuple[int, int], 
                       actions: List[str]) -> List[Tuple[int, int]]:
        """
        Convert action sequence to path positions.
        
        Args:
            start: Starting position
            actions: List of actions ('UP', 'DOWN', 'LEFT', 'RIGHT')
            
        Returns:
            List of positions visited
        """
        action_map = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }
        
        path = [start]
        current = start
        
        for action in actions:
            if action not in action_map:
                continue
                
            dr, dc = action_map[action]
            new_pos = (current[0] + dr, current[1] + dc)
            
            # Check if valid move
            if (0 <= new_pos[0] < self.config.height and
                0 <= new_pos[1] < self.config.width and
                self.grid[new_pos] != 0):
                current = new_pos
                path.append(current)
        
        return path
    
    def path_to_actions(self, path: List[Tuple[int, int]]) -> List[str]:
        """
        Convert path positions to action sequence.
        
        Args:
            path: List of (row, col) positions
            
        Returns:
            List of actions ('UP', 'DOWN', 'LEFT', 'RIGHT')
        """
        actions = []
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            dr = next_pos[0] - current[0]
            dc = next_pos[1] - current[1]
            
            if dr == -1:
                actions.append('UP')
            elif dr == 1:
                actions.append('DOWN')
            elif dc == -1:
                actions.append('LEFT')
            elif dc == 1:
                actions.append('RIGHT')
        
        return actions


class MazeDataset:
    """
    Generate a dataset of mazes with solutions for training.
    """
    
    def __init__(self, num_mazes: int, config: MazeConfig):
        self.config = config
        self.mazes = []
        self.solutions = []
        
        for i in range(num_mazes):
            maze = Maze(MazeConfig(
                height=config.height,
                width=config.width,
                wall_probability=config.wall_probability,
                ensure_solvable=config.ensure_solvable,
                seed=config.seed + i if config.seed else None
            ))
            
            solution = maze.solve()
            if solution:
                self.mazes.append(maze)
                self.solutions.append(solution)
    
    def get_training_pairs(self) -> List[Tuple[Maze, List[str]]]:
        """
        Get (maze, action_sequence) pairs for training.
        
        Returns:
            List of (maze, actions) tuples
        """
        pairs = []
        for maze, solution in zip(self.mazes, self.solutions):
            actions = maze.path_to_actions(solution)
            pairs.append((maze, actions))
        return pairs
    
    def __len__(self):
        return len(self.mazes)
    
    def __getitem__(self, idx):
        return self.mazes[idx], self.solutions[idx]


def generate_simple_maze(size: int = 10, seed: Optional[int] = None) -> Maze:
    """
    Convenience function to generate a simple square maze.
    
    Args:
        size: Size of the square maze
        seed: Random seed for reproducibility
        
    Returns:
        Maze instance
    """
    config = MazeConfig(
        height=size,
        width=size,
        wall_probability=0.25,
        ensure_solvable=True,
        seed=seed
    )
    return Maze(config)


if __name__ == "__main__":
    # Demo usage
    print("Generating a simple maze...")
    maze = generate_simple_maze(size=15, seed=42)
    
    print("\nMaze without solution:")
    print(maze.to_text())
    
    print("\nSolving maze...")
    solution = maze.solve()
    
    if solution:
        print(f"\nSolution found! Path length: {len(solution)}")
        print("\nMaze with solution marked:")
        print(maze.to_text(show_solution=True))
        
        print("\nSolution as actions:")
        actions = maze.path_to_actions(solution)
        print(' -> '.join(actions))
    else:
        print("\nNo solution found!")
