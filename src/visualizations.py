"""
Visualization utilities for maze environments and model behavior.

Provides plotting functions for:
- Maze visualization
- Attention heatmaps
- Training curves
- Hidden state evolution
- Comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Optional, Dict
import seaborn as sns


class MazeVisualizer:
    """Visualize mazes and solutions"""
    
    # Color scheme
    COLORS = {
        'wall': '#2C3E50',      # Dark blue-gray
        'path': '#ECF0F1',      # Light gray
        'start': '#27AE60',     # Green
        'goal': '#E74C3C',      # Red
        'solution': '#F39C12',  # Orange
        'visited': '#3498DB',   # Blue
    }
    
    @staticmethod
    def plot_maze(maze, ax=None, show_solution=False, title="Maze"):
        """
        Plot a maze with optional solution overlay.
        
        Args:
            maze: Maze instance
            ax: Matplotlib axis (creates new if None)
            show_solution: Whether to show solution path
            title: Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create color-coded grid
        display_grid = maze.grid.copy().astype(float)
        
        # Mark solution if requested
        if show_solution and maze.solution:
            for pos in maze.solution[1:-1]:  # Don't overwrite start/goal
                if maze.grid[pos] == 1:  # Only mark path cells
                    display_grid[pos] = 4  # Solution marker
        
        # Color map: wall, path, start, goal, solution
        colors = ['#2C3E50', '#ECF0F1', '#27AE60', '#E74C3C', '#F39C12']
        cmap = ListedColormap(colors)
        
        # Plot
        im = ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=4)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, maze.config.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.config.height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
        # Remove major ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#2C3E50', label='Wall'),
            mpatches.Patch(color='#ECF0F1', label='Path'),
            mpatches.Patch(color='#27AE60', label='Start'),
            mpatches.Patch(color='#E74C3C', label='Goal'),
        ]
        if show_solution:
            legend_elements.append(
                mpatches.Patch(color='#F39C12', label='Solution')
            )
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.05, 1), fontsize=10)
        
        return ax
    
    @staticmethod
    def plot_maze_comparison(maze, predicted_path=None, true_path=None, 
                           figsize=(15, 5)):
        """
        Plot maze with predicted and true paths side by side.
        
        Args:
            maze: Maze instance
            predicted_path: List of positions from model prediction
            true_path: List of positions from ground truth
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original maze
        MazeVisualizer.plot_maze(maze, ax=axes[0], title="Original Maze")
        
        # Predicted path
        if predicted_path:
            display_grid = maze.grid.copy().astype(float)
            for pos in predicted_path:
                if maze.grid[pos] != 2 and maze.grid[pos] != 3:
                    display_grid[pos] = 4
            
            colors = ['#2C3E50', '#ECF0F1', '#27AE60', '#E74C3C', '#3498DB']
            cmap = ListedColormap(colors)
            axes[1].imshow(display_grid, cmap=cmap, vmin=0, vmax=4)
            axes[1].set_title("Predicted Path", fontsize=14, fontweight='bold')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        # True path
        MazeVisualizer.plot_maze(maze, ax=axes[2], show_solution=True, 
                               title="True Solution")
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def animate_path(maze, path, save_path=None):
        """
        Create an animation showing path exploration.
        (Simplified version - full animation would use matplotlib.animation)
        """
        fig, axes = plt.subplots(1, min(len(path), 6), figsize=(20, 4))
        if len(path) < 6:
            axes = [axes] if len(path) == 1 else list(axes)
        
        step_indices = np.linspace(0, len(path)-1, min(6, len(path)), dtype=int)
        
        for idx, step in enumerate(step_indices):
            display_grid = maze.grid.copy().astype(float)
            
            # Mark visited path
            for pos in path[:step+1]:
                if maze.grid[pos] == 1:
                    display_grid[pos] = 4
            
            colors = ['#2C3E50', '#ECF0F1', '#27AE60', '#E74C3C', '#F39C12']
            cmap = ListedColormap(colors)
            
            ax = axes[idx] if len(path) > 1 else axes[0]
            ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=4)
            ax.set_title(f"Step {step}/{len(path)-1}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        return fig


class AttentionVisualizer:
    """Visualize attention patterns"""
    
    @staticmethod
    def plot_attention_heatmap(attention_weights, source_labels=None, 
                              target_labels=None, ax=None, title="Attention Weights"):
        """
        Plot attention weight heatmap.
        
        Args:
            attention_weights: 2D array of attention weights [target, source]
            source_labels: Labels for source positions
            target_labels: Labels for target positions
            ax: Matplotlib axis
            title: Plot title
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(attention_weights, ax=ax, cmap='YlOrRd', 
                   cbar_kws={'label': 'Attention Weight'},
                   xticklabels=source_labels if source_labels else 'auto',
                   yticklabels=target_labels if target_labels else 'auto')
        
        ax.set_xlabel('Source Position', fontsize=12)
        ax.set_ylabel('Target Position', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return ax
    
    @staticmethod
    def plot_multi_head_attention(attention_weights_list, num_heads=None, 
                                 figsize=(15, 10)):
        """
        Plot attention patterns from multiple heads.
        
        Args:
            attention_weights_list: List of attention weight matrices
            num_heads: Number of heads to display (uses all if None)
            figsize: Figure size
        """
        if num_heads is None:
            num_heads = len(attention_weights_list)
        
        n_cols = min(4, num_heads)
        n_rows = (num_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for idx, (weights, ax) in enumerate(zip(attention_weights_list[:num_heads], 
                                                axes)):
            sns.heatmap(weights, ax=ax, cmap='YlOrRd', 
                       cbar=True, square=True)
            ax.set_title(f'Head {idx+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Source')
            ax.set_ylabel('Target')
        
        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_attention_flow(attention_weights, positions, ax=None):
        """
        Plot attention as flow/arrows between positions.
        
        Args:
            attention_weights: 2D attention matrix
            positions: 2D array of position coordinates
            ax: Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot positions
        ax.scatter(positions[:, 0], positions[:, 1], s=100, c='blue', 
                  alpha=0.6, zorder=3)
        
        # Plot attention flows
        threshold = 0.1  # Only show significant attention
        for i in range(len(positions)):
            for j in range(len(positions)):
                weight = attention_weights[i, j]
                if weight > threshold and i != j:
                    ax.arrow(positions[j, 0], positions[j, 1],
                           positions[i, 0] - positions[j, 0],
                           positions[i, 1] - positions[j, 1],
                           alpha=weight, width=weight*0.02,
                           head_width=0.3, head_length=0.2,
                           fc='red', ec='red', zorder=2)
        
        ax.set_title('Attention Flow', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        return ax


class TrainingVisualizer:
    """Visualize training progress and model behavior"""
    
    @staticmethod
    def plot_training_curves(history: Dict[str, List], figsize=(15, 5)):
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history:
            axes[2].plot(history['lr'], linewidth=2, color='green')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_hidden_state_evolution(hidden_states: np.ndarray, 
                                   labels: Optional[List[str]] = None,
                                   figsize=(12, 6)):
        """
        Plot evolution of hidden state values over sequence.
        
        Args:
            hidden_states: Array of shape [sequence_length, hidden_dim]
            labels: Optional labels for time steps
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Heatmap of all dimensions
        sns.heatmap(hidden_states.T, ax=axes[0], cmap='coolwarm', 
                   center=0, cbar_kws={'label': 'Activation'})
        axes[0].set_xlabel('Sequence Position')
        axes[0].set_ylabel('Hidden Dimension')
        axes[0].set_title('Hidden State Evolution', fontsize=12, fontweight='bold')
        
        # Line plot of selected dimensions
        seq_len = hidden_states.shape[0]
        dims_to_plot = min(5, hidden_states.shape[1])
        for dim in range(dims_to_plot):
            axes[1].plot(hidden_states[:, dim], 
                        label=f'Dim {dim}', linewidth=2, alpha=0.7)
        
        axes[1].set_xlabel('Sequence Position')
        axes[1].set_ylabel('Activation Value')
        axes[1].set_title('Selected Dimensions', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_performance_vs_length(lengths: List[int], 
                                  accuracies: List[float],
                                  model_names: List[str],
                                  figsize=(10, 6)):
        """
        Plot model performance vs sequence length.
        
        Args:
            lengths: List of sequence lengths
            accuracies: List of accuracy values for each model
            model_names: Names of models
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, acc_list in zip(model_names, accuracies):
            ax.plot(lengths, acc_list, marker='o', linewidth=2, 
                   markersize=8, label=model_name)
        
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Performance vs Sequence Length', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


class ComparisonVisualizer:
    """Compare different models and approaches"""
    
    @staticmethod
    def plot_model_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                            figsize=(12, 5)):
        """
        Plot comparison of multiple models across metrics.
        
        Args:
            metrics_dict: {model_name: {metric_name: value}}
            figsize: Figure size
        """
        model_names = list(metrics_dict.keys())
        metric_names = list(next(iter(metrics_dict.values())).keys())
        
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metric_names):
            values = [metrics_dict[model][metric] for model in model_names]
            
            bars = axes[idx].bar(range(len(model_names)), values, 
                               color=['#3498DB', '#E74C3C', '#F39C12', '#27AE60'][:len(model_names)])
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(metric.capitalize(), fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}',
                             ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig, axes


def set_style():
    """Set consistent matplotlib style for all plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


if __name__ == "__main__":
    # Demo visualizations
    from maze_envs import generate_simple_maze
    
    set_style()
    
    print("Generating demo visualizations...")
    
    # Create a maze
    maze = generate_simple_maze(size=15, seed=42)
    maze.solve()
    
    # Plot maze
    fig, ax = plt.subplots(figsize=(8, 8))
    MazeVisualizer.plot_maze(maze, ax=ax, show_solution=True)
    plt.savefig('/home/claude/transformer-maze/demo_maze.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_maze.png")
    
    # Plot attention heatmap (random for demo)
    attention = np.random.rand(10, 10)
    attention = attention / attention.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    AttentionVisualizer.plot_attention_heatmap(attention, ax=ax)
    plt.savefig('/home/claude/transformer-maze/demo_attention.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_attention.png")
    
    print("\nDemo complete!")
