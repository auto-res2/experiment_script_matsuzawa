"""
Utility functions for visualization.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def save_plot(
    x, y, title, xlabel, ylabel, 
    filename, 
    color='blue', 
    marker='o', 
    linestyle='-',
    legend=None,
    grid=True,
    figsize=(8, 6),
    dpi=300
):
    """
    Save a plot as a high-quality PDF.
    
    Args:
        x: Data for x-axis
        y: Data for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
        color: Line color
        marker: Point marker style
        linestyle: Line style
        legend: Legend labels (if applicable)
        grid: Whether to show grid
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    if isinstance(y, list) and isinstance(y[0], (list, np.ndarray, tuple)):
        for i, yi in enumerate(y):
            plt.plot(x, yi, marker=marker, linestyle=linestyle, 
                     color=color[i] if isinstance(color, list) else color)
    else:
        plt.plot(x, y, marker=marker, linestyle=linestyle, color=color)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if legend:
        plt.legend(legend)
    
    if grid:
        plt.grid(True)
    
    plt.savefig(filename, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close()
    
    print(f"Plot saved as {filename}")
