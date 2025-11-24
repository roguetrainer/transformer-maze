"""
RNN-based maze solver

Demonstrates sequential processing approach where the model must process
the maze step-by-step, maintaining hidden state as it goes.

This module implements:
- Simple RNN
- LSTM (improved memory)
- GRU (efficient variant)

All for the task of generating solution paths through mazes.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class RNNConfig:
    """Configuration for RNN models"""
    input_dim: int = 64  # Maze encoding dimension
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    cell_type: str = 'lstm'  # 'rnn', 'lstm', or 'gru'


class MazeEncoder(nn.Module):
    """
    Encode maze state into a fixed-size representation.
    
    The maze is flattened into a sequence where each position is encoded
    based on its type (wall, path, start, goal).
    """
    
    def __init__(self, maze_size: int, embed_dim: int = 64):
        super().__init__()
        self.maze_size = maze_size
        self.embed_dim = embed_dim
        
        # Embedding for each cell type
        # 0=wall, 1=path, 2=start, 3=goal
        self.cell_embedding = nn.Embedding(4, embed_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Embedding(maze_size * maze_size, embed_dim)
        
    def forward(self, maze_grid):
        """
        Encode maze grid.
        
        Args:
            maze_grid: [batch_size, height, width] tensor of cell types
            
        Returns:
            [batch_size, height*width, embed_dim] encoded maze
        """
        batch_size, height, width = maze_grid.shape
        
        # Flatten spatial dimensions
        flat_maze = maze_grid.view(batch_size, -1)  # [B, H*W]
        
        # Embed cell types
        cell_embeds = self.cell_embedding(flat_maze)  # [B, H*W, D]
        
        # Add positional information
        positions = torch.arange(height * width, device=maze_grid.device)
        pos_embeds = self.pos_embedding(positions)  # [H*W, D]
        
        # Combine
        encoded = cell_embeds + pos_embeds.unsqueeze(0)  # [B, H*W, D]
        
        return encoded


class ActionDecoder(nn.Module):
    """
    Decode hidden state into action probabilities.
    
    Actions: UP=0, DOWN=1, LEFT=2, RIGHT=3, STOP=4
    """
    
    def __init__(self, hidden_dim: int, num_actions: int = 5):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self, hidden_state):
        """
        Args:
            hidden_state: [batch_size, hidden_dim]
            
        Returns:
            [batch_size, num_actions] logits
        """
        return self.decoder(hidden_state)


class RNNMazeSolver(nn.Module):
    """
    RNN-based maze solver using sequential processing.
    
    The model:
    1. Encodes the maze into embeddings
    2. Processes current position sequentially through RNN
    3. Predicts next action at each step
    4. Continues until reaching goal or max steps
    """
    
    def __init__(self, config: RNNConfig, maze_size: int = 15):
        super().__init__()
        self.config = config
        self.maze_size = maze_size
        
        # Maze encoder
        self.maze_encoder = MazeEncoder(maze_size, config.input_dim)
        
        # Position embedding (current position in maze)
        self.position_embed = nn.Embedding(maze_size * maze_size, config.input_dim)
        
        # RNN core
        if config.cell_type == 'rnn':
            self.rnn = nn.RNN(
                config.input_dim * 2,  # maze + position
                config.hidden_dim,
                config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0
            )
        elif config.cell_type == 'lstm':
            self.rnn = nn.LSTM(
                config.input_dim * 2,
                config.hidden_dim,
                config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0
            )
        elif config.cell_type == 'gru':
            self.rnn = nn.GRU(
                config.input_dim * 2,
                config.hidden_dim,
                config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown cell type: {config.cell_type}")
        
        # Action decoder
        self.action_decoder = ActionDecoder(config.hidden_dim)
        
    def encode_position(self, position: torch.Tensor):
        """
        Encode current position in maze.
        
        Args:
            position: [batch_size, 2] tensor of (row, col) positions
            
        Returns:
            [batch_size, input_dim] position embedding
        """
        # Convert (row, col) to flat index
        flat_pos = position[:, 0] * self.maze_size + position[:, 1]
        return self.position_embed(flat_pos)
    
    def forward(self, maze_grid, position_sequence, hidden=None):
        """
        Forward pass through the RNN.
        
        Args:
            maze_grid: [batch_size, height, width] maze representation
            position_sequence: [batch_size, seq_len, 2] sequence of positions
            hidden: Optional initial hidden state
            
        Returns:
            action_logits: [batch_size, seq_len, num_actions]
            hidden: Final hidden state
        """
        batch_size, seq_len, _ = position_sequence.shape
        
        # Encode maze once (it doesn't change)
        maze_encoding = self.maze_encoder(maze_grid)  # [B, H*W, D]
        
        # Use mean pooling to get global maze context
        maze_context = maze_encoding.mean(dim=1)  # [B, D]
        maze_context = maze_context.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, D]
        
        # Encode position sequence
        position_embeds = []
        for t in range(seq_len):
            pos = position_sequence[:, t, :]  # [B, 2]
            pos_embed = self.encode_position(pos)  # [B, D]
            position_embeds.append(pos_embed)
        position_embeds = torch.stack(position_embeds, dim=1)  # [B, T, D]
        
        # Combine maze context and position
        rnn_input = torch.cat([maze_context, position_embeds], dim=-1)  # [B, T, 2D]
        
        # Pass through RNN
        rnn_output, hidden = self.rnn(rnn_input, hidden)  # [B, T, H]
        
        # Decode to actions
        action_logits = self.action_decoder(rnn_output)  # [B, T, num_actions]
        
        return action_logits, hidden
    
    def generate_path(self, maze_grid, start_pos, max_steps=100):
        """
        Generate a path through the maze autoregressively.
        
        Args:
            maze_grid: [1, height, width] single maze
            start_pos: (row, col) starting position
            max_steps: Maximum steps to generate
            
        Returns:
            List of (row, col) positions
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convert inputs to tensors
        if not isinstance(maze_grid, torch.Tensor):
            maze_grid = torch.tensor(maze_grid, dtype=torch.long, device=device)
        if maze_grid.dim() == 2:
            maze_grid = maze_grid.unsqueeze(0)
        
        current_pos = torch.tensor([[start_pos[0], start_pos[1]]], 
                                   dtype=torch.long, device=device)
        
        path = [start_pos]
        hidden = None
        
        # Action to movement mapping
        action_to_delta = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
            4: (0, 0),   # STOP
        }
        
        with torch.no_grad():
            for step in range(max_steps):
                # Forward pass
                action_logits, hidden = self.forward(
                    maze_grid, 
                    current_pos.unsqueeze(1),  # [1, 1, 2]
                    hidden
                )
                
                # Sample action
                action_probs = torch.softmax(action_logits[0, 0], dim=0)
                action = torch.multinomial(action_probs, 1).item()
                
                # Check if STOP
                if action == 4:
                    break
                
                # Update position
                dr, dc = action_to_delta[action]
                new_row = current_pos[0, 0].item() + dr
                new_col = current_pos[0, 1].item() + dc
                
                # Check bounds and walls
                if (0 <= new_row < self.maze_size and 
                    0 <= new_col < self.maze_size and
                    maze_grid[0, new_row, new_col] != 0):  # Not a wall
                    
                    current_pos = torch.tensor([[new_row, new_col]], 
                                              dtype=torch.long, device=device)
                    path.append((new_row, new_col))
                    
                    # Check if reached goal
                    if maze_grid[0, new_row, new_col] == 3:
                        break
        
        return path
    
    def get_hidden_states(self, maze_grid, position_sequence):
        """
        Get hidden state evolution for analysis.
        
        Args:
            maze_grid: [batch_size, height, width]
            position_sequence: [batch_size, seq_len, 2]
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_dim]
        """
        self.eval()
        with torch.no_grad():
            _, _ = self.forward(maze_grid, position_sequence)
            # For LSTM, extract hidden state from tuple
            # For analysis, we'd need to modify forward to return intermediate states
        # This is a placeholder - full implementation would store intermediate states
        return None


class RNNTrainer:
    """
    Trainer for RNN maze solver.
    """
    
    def __init__(self, model: RNNMazeSolver, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, maze_grids, position_sequences, action_sequences):
        """
        Single training step.
        
        Args:
            maze_grids: [batch_size, height, width]
            position_sequences: [batch_size, seq_len, 2]
            action_sequences: [batch_size, seq_len] target actions
            
        Returns:
            loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        action_logits, _ = self.model(maze_grids, position_sequences)
        
        # Reshape for loss computation
        batch_size, seq_len, num_actions = action_logits.shape
        action_logits = action_logits.reshape(-1, num_actions)
        action_sequences = action_sequences.reshape(-1)
        
        # Compute loss
        loss = self.criterion(action_logits, action_sequences)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, maze_grids, position_sequences, action_sequences):
        """
        Evaluate model on validation data.
        
        Returns:
            loss, accuracy
        """
        self.model.eval()
        with torch.no_grad():
            action_logits, _ = self.model(maze_grids, position_sequences)
            
            batch_size, seq_len, num_actions = action_logits.shape
            action_logits = action_logits.reshape(-1, num_actions)
            action_sequences = action_sequences.reshape(-1)
            
            loss = self.criterion(action_logits, action_sequences)
            
            predictions = action_logits.argmax(dim=-1)
            accuracy = (predictions == action_sequences).float().mean()
        
        return loss.item(), accuracy.item()


def create_simple_rnn(maze_size: int = 15, hidden_dim: int = 128) -> RNNMazeSolver:
    """Convenience function to create a simple RNN model"""
    config = RNNConfig(
        input_dim=64,
        hidden_dim=hidden_dim,
        num_layers=1,
        cell_type='rnn'
    )
    return RNNMazeSolver(config, maze_size)


def create_lstm(maze_size: int = 15, hidden_dim: int = 128) -> RNNMazeSolver:
    """Convenience function to create an LSTM model"""
    config = RNNConfig(
        input_dim=64,
        hidden_dim=hidden_dim,
        num_layers=2,
        cell_type='lstm'
    )
    return RNNMazeSolver(config, maze_size)


if __name__ == "__main__":
    print("Testing RNN Maze Solver...")
    
    # Create model
    model = create_lstm(maze_size=15, hidden_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    maze_size = 15
    
    dummy_maze = torch.randint(0, 4, (batch_size, maze_size, maze_size))
    dummy_positions = torch.randint(0, maze_size, (batch_size, seq_len, 2))
    
    action_logits, hidden = model(dummy_maze, dummy_positions)
    print(f"Output shape: {action_logits.shape}")
    print(f"Hidden state type: {type(hidden)}")
    
    print("\nRNN Maze Solver ready!")
