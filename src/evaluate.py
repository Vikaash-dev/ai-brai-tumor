"""
Evaluation Module for Brain Tumor Detection.

This module provides functions for evaluating trained models,
computing metrics, and generating evaluation reports.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np

# TensorFlow imports
import tensorflow as tf

# Metrics
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import create_test_generator, load_images_from_directory
from models.cnn_model import load_model


def evaluate_model(
    model: tf.keras.Model,
    test_dir: str = config.TEST_DIR,
    batch_size: int = config.BATCH_SIZE,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained Keras model
        test_dir: Path to test data directory
        batch_size: Batch size for evaluation
        verbose: Verbosity level

    Returns:
        Dictionary containing evaluation metrics
    """
    print("=" * 60)
    print("Brain Tumor Detection - Model Evaluation")
    print("=" * 60)

    # Create test generator
    test_generator = create_test_generator(test_dir, batch_size=batch_size)
    print(f"\nTest samples: {test_generator.samples}")

    # Get predictions
    print("Generating predictions...")
    predictions = model.predict(test_generator, verbose=verbose)
    y_pred_proba = predictions
    y_pred = np.argmax(predictions, axis=1)

    # Get true labels
    y_true = test_generator.classes

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba[:, 1])

    # Print report
    print_evaluation_report(metrics, config.CLASS_NAMES)

    # Save report to file
    save_evaluation_report(metrics, config.CLASS_NAMES)

    return metrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities for positive class

    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=config.CLASS_NAMES,
            output_dict=True
        )
    }

    # ROC-AUC (only for binary classification)
    if len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}

    return metrics


def print_evaluation_report(metrics: Dict[str, Any], class_names: list) -> None:
    """
    Print a formatted evaluation report.

    Args:
        metrics: Dictionary of evaluation metrics
        class_names: List of class names
    """
    print("\n" + "=" * 40)
    print("EVALUATION METRICS")
    print("=" * 40)

    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\n" + "-" * 40)
    print("CONFUSION MATRIX")
    print("-" * 40)
    cm = metrics['confusion_matrix']
    print(f"\n{'':15} Predicted")
    print(f"{'':15} {class_names[0]:>10} {class_names[1]:>10}")
    print(f"Actual {class_names[0]:>8} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"       {class_names[1]:>8} {cm[1][0]:>10} {cm[1][1]:>10}")

    print("\n" + "-" * 40)
    print("CLASSIFICATION REPORT")
    print("-" * 40)
    report = metrics['classification_report']
    print(f"\n{'Class':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 55)
    for class_name in class_names:
        if class_name in report:
            r = report[class_name]
            print(f"{class_name:15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                  f"{r['f1-score']:>10.4f} {r['support']:>10.0f}")


def save_evaluation_report(metrics: Dict[str, Any], class_names: list) -> None:
    """
    Save evaluation report to a text file.

    Args:
        metrics: Dictionary of evaluation metrics
        class_names: List of class names
    """
    report_path = os.path.join(config.RESULTS_DIR, 'classification_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BRAIN TUMOR DETECTION - EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")

        if 'roc_auc' in metrics:
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")

        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"                    Predicted\n")
        f.write(f"                    {class_names[0]:>10} {class_names[1]:>10}\n")
        f.write(f"Actual {class_names[0]:>12} {cm[0][0]:>10} {cm[0][1]:>10}\n")
        f.write(f"       {class_names[1]:>12} {cm[1][0]:>10} {cm[1][1]:>10}\n")

        f.write("\n\nPER-CLASS METRICS\n")
        f.write("-" * 40 + "\n")
        report = metrics['classification_report']
        f.write(f"{'Class':15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        for class_name in class_names:
            if class_name in report:
                r = report[class_name]
                f.write(f"{class_name:15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
                       f"{r['f1-score']:>10.4f} {r['support']:>10.0f}\n")

    print(f"\nEvaluation report saved to: {report_path}")


def evaluate_from_file(
    model_path: str,
    test_dir: str = config.TEST_DIR,
    batch_size: int = config.BATCH_SIZE
) -> Dict[str, Any]:
    """
    Load a model from file and evaluate it.

    Args:
        model_path: Path to saved model file
        test_dir: Path to test data directory
        batch_size: Batch size for evaluation

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    return evaluate_model(model, test_dir, batch_size)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate brain tumor detection model')
    parser.add_argument('--model-path', type=str, default=config.BEST_MODEL_PATH,
                       help='Path to saved model')
    parser.add_argument('--test-dir', type=str, default=config.TEST_DIR,
                       help='Path to test data directory')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')

    args = parser.parse_args()

    if os.path.exists(args.model_path):
        metrics = evaluate_from_file(
            args.model_path,
            args.test_dir,
            args.batch_size
        )
    else:
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using: python -m src.train")
