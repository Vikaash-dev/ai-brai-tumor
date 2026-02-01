#!/usr/bin/env python
"""
One-Click Training and Testing Script for Brain Tumor Detection.

This script provides a simple interface to train and test the brain tumor
detection model with various configurations.
"""

import os
import sys
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def main():
    """Main entry point for training and testing."""
    parser = argparse.ArgumentParser(
        description='Brain Tumor Detection - One-Click Training and Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python one_click_train_test.py --mode train

  # Train with specific model type
  python one_click_train_test.py --mode train --model-type regularized --epochs 100

  # Evaluate trained model
  python one_click_train_test.py --mode evaluate

  # Make prediction on single image
  python one_click_train_test.py --mode predict --image path/to/image.jpg

  # Full pipeline: train and evaluate
  python one_click_train_test.py --mode full
        """
    )

    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'evaluate', 'predict', 'full'],
                       help='Operation mode')
    parser.add_argument('--model-type', type=str, default='baseline',
                       choices=['baseline', 'simple', 'regularized'],
                       help='Type of CNN model to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--model-path', type=str, default=config.BEST_MODEL_PATH,
                       help='Path to model for evaluation/prediction')
    parser.add_argument('--image', type=str,
                       help='Path to image for single prediction')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation during training')
    parser.add_argument('--deduplicate', action='store_true',
                       help='Run data deduplication before training')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')

    args = parser.parse_args()

    print("=" * 70)
    print("BRAIN TUMOR DETECTION - PHOENIX PROTOCOL")
    print("=" * 70)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Model Type: {args.model_type}")
    print(f"Model Path: {args.model_path}")

    if args.mode == 'train' or args.mode == 'full':
        train_mode(args)

    if args.mode == 'evaluate' or args.mode == 'full':
        evaluate_mode(args)

    if args.mode == 'predict':
        predict_mode(args)

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)


def train_mode(args):
    """Run training."""
    print("\n" + "-" * 50)
    print("TRAINING MODE")
    print("-" * 50)

    from src.train import train_model
    from src.visualize import plot_training_history

    model, history = train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        augment_train=not args.no_augment
    )

    if args.visualize:
        plot_training_history(
            history,
            save_path=os.path.join(config.RESULTS_DIR, 'training_history.png'),
            show=False
        )

    print(f"\nTraining completed!")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")


def evaluate_mode(args):
    """Run evaluation."""
    print("\n" + "-" * 50)
    print("EVALUATION MODE")
    print("-" * 50)

    from src.evaluate import evaluate_from_file
    from src.visualize import visualize_evaluation_results

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first with: python one_click_train_test.py --mode train")
        return

    metrics = evaluate_from_file(args.model_path)

    if args.visualize:
        visualize_evaluation_results(metrics, output_dir=config.RESULTS_DIR, show=False)

    print(f"\nEvaluation completed!")
    print(f"Results saved to: {config.RESULTS_DIR}")


def predict_mode(args):
    """Run prediction."""
    print("\n" + "-" * 50)
    print("PREDICTION MODE")
    print("-" * 50)

    from src.predict import predict_single_image, print_prediction, interactive_prediction
    from models.cnn_model import load_model

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first with: python one_click_train_test.py --mode train")
        return

    model = load_model(args.model_path)

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
        result = predict_single_image(model, args.image)
        print_prediction(result)
    else:
        interactive_prediction(model)


if __name__ == '__main__':
    main()
