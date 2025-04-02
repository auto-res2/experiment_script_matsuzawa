"""
Utility functions for visualization in the MEAB-DG experiments.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_plot_style():
    """Set up plot style for publication-quality figures."""
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["figure.titlesize"] = 20

def save_plot(filename, directory="logs", tight_layout=True, formats=["pdf"]):
    """
    Save plot in high-quality format(s) suitable for academic papers.
    
    Args:
        filename: Base filename without extension
        directory: Directory to save the plot
        tight_layout: Whether to use tight layout
        formats: List of formats to save (default: ["pdf"])
    """
    if tight_layout:
        plt.tight_layout()
    
    for fmt in formats:
        save_path = f"{directory}/{filename}.{fmt}"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

def plot_training_curve(epochs, train_losses, val_metrics, metric_name="Accuracy", 
                       title="Training Curve", filename="training_curve"):
    """
    Plot training loss and validation metric curves.
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses for each epoch
        val_metrics: List of validation metrics for each epoch
        metric_name: Name of the validation metric (default: "Accuracy")
        title: Plot title
        filename: Base filename to save the plot
    """
    setup_plot_style()
    fig, ax1 = plt.subplots()
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    line1 = ax1.plot(epochs, train_losses, "o-", color="tab:blue", label="Training Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(metric_name, color="tab:green")
    line2 = ax2.plot(epochs, val_metrics, "s-", color="tab:green", label=f"Validation {metric_name}")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")
    
    plt.title(title)
    save_plot(filename)
    plt.close()

def plot_comparison_bar(categories, values, title="Comparison", 
                      xlabel="Categories", ylabel="Value", filename="comparison_bar"):
    """
    Create a bar plot comparing values across categories.
    
    Args:
        categories: List of category names
        values: List of values for each category
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Base filename to save the plot
    """
    setup_plot_style()
    plt.figure()
    sns.barplot(x=categories, y=values, palette="Blues_d")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_plot(filename)
    plt.close()

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", filename="confusion_matrix"):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        classes: List of class names
        title: Plot title
        filename: Base filename to save the plot
    """
    setup_plot_style()
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
               xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    save_plot(filename)
    plt.close()
