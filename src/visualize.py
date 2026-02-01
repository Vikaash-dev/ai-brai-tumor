"""
Visualization Module for Brain Tumor Detection.

This module provides functions for visualizing training history,
model performance, and predictions.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Training history dictionary
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss', color='blue', linewidth=2)
    axes[1].plot(history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str] = config.CLASS_NAMES,
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: bool = False
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 14}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the ROC curve
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str] = config.CLASS_NAMES,
    num_samples: int = 9,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot a grid of sample predictions.

    Args:
        images: Array of images
        true_labels: Array of true labels
        predictions: Array of prediction probabilities
        class_names: List of class names
        num_samples: Number of samples to display
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    num_samples = min(num_samples, len(images))
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i])

        pred_idx = np.argmax(predictions[i])
        pred_class = class_names[pred_idx]
        true_class = class_names[true_labels[i]]
        confidence = predictions[i][pred_idx]

        color = 'green' if pred_class == true_class else 'red'
        ax.set_title(f'True: {true_class}\nPred: {pred_class} ({confidence:.2%})',
                    color=color, fontsize=10)
        ax.axis('off')

    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = 'Class Distribution',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot class distribution as a bar chart.

    Args:
        class_counts: Dictionary mapping class names to counts
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(8, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bars = plt.bar(classes, counts, color=['#3498db', '#e74c3c'])

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=12)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_evaluation_results(
    metrics: Dict[str, Any],
    history: Optional[Dict[str, list]] = None,
    output_dir: str = config.RESULTS_DIR,
    show: bool = False
) -> None:
    """
    Generate and save all visualization plots for evaluation results.

    Args:
        metrics: Dictionary of evaluation metrics
        history: Training history dictionary (optional)
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot training history if available
    if history:
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, 'training_history.png'),
            show=show
        )

    # Plot confusion matrix
    if 'confusion_matrix' in metrics:
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=os.path.join(output_dir, 'confusion_matrix.png'),
            show=show
        )

        # Also plot normalized version
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            save_path=os.path.join(output_dir, 'confusion_matrix_normalized.png'),
            show=show,
            normalize=True
        )

    # Plot ROC curve if available
    if 'roc_curve' in metrics and 'roc_auc' in metrics:
        plot_roc_curve(
            metrics['roc_curve']['fpr'],
            metrics['roc_curve']['tpr'],
            metrics['roc_auc'],
            save_path=os.path.join(output_dir, 'roc_curve.png'),
            show=show
        )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    # Demo visualization with sample data
    print("Visualization module - Demo mode")

    # Generate sample training history with realistic progression
    epochs = 50
    sample_history = {
        'accuracy': np.linspace(0.7, 0.95, epochs) + np.random.uniform(-0.02, 0.02, epochs),
        'val_accuracy': np.linspace(0.65, 0.92, epochs) + np.random.uniform(-0.03, 0.03, epochs),
        'loss': np.linspace(1.5, 0.2, epochs) + np.random.uniform(-0.1, 0.1, epochs),
        'val_loss': np.linspace(1.6, 0.3, epochs) + np.random.uniform(-0.15, 0.15, epochs)
    }

    # Generate sample confusion matrix
    sample_cm = np.array([[85, 15], [10, 90]])

    print("Generating sample visualizations...")

    # Plot training history
    plot_training_history(
        sample_history,
        save_path=os.path.join(config.RESULTS_DIR, 'sample_training_history.png'),
        show=False
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        sample_cm,
        save_path=os.path.join(config.RESULTS_DIR, 'sample_confusion_matrix.png'),
        show=False
    )

    # Plot class distribution
    sample_distribution = {'no_tumor': 1500, 'tumor': 1200}
    plot_class_distribution(
        sample_distribution,
        save_path=os.path.join(config.RESULTS_DIR, 'sample_class_distribution.png'),
        show=False
    )

    print("Sample visualizations saved to results directory.")
