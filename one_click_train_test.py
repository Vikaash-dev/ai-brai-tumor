#!/usr/bin/env python
"""
Phoenix Protocol - One-Click SOTA Brain Tumor Detection.

This script provides a unified interface to the SOTA brain tumor detection system
featuring NeuroSnake architecture with Dynamic Snake Convolutions, Coordinate
Attention, Physics-Informed Augmentation, and Adan Optimizer.

Usage:
    # Train SOTA model (NeuroSnake + Coordinate Attention)
    python one_click_train_test.py --mode train --model-type neurosnake_ca

    # Train with data deduplication (prevents data leakage)
    python one_click_train_test.py --mode train --deduplicate

    # Full pipeline: train, evaluate, visualize
    python one_click_train_test.py --mode full --visualize
"""

import os
import sys
import argparse
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


# Available model types
MODEL_TYPES = {
    'baseline': 'Baseline CNN (standard convolutions)',
    'simple': 'Simple CNN (lightweight)',
    'regularized': 'Regularized CNN (L2 regularization)',
    'neurosnake': 'NeuroSnake (Dynamic Snake Conv + MobileViT)',
    'neurosnake_ca': 'NeuroSnake + Coordinate Attention [SOTA]',
}


def main():
    """Main entry point for Phoenix Protocol training and testing."""
    parser = argparse.ArgumentParser(
        description='Phoenix Protocol - SOTA Brain Tumor Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════════════╗
║                    PHOENIX PROTOCOL - EXAMPLES                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  # Train SOTA model (recommended)                                     ║
║  python one_click_train_test.py --mode train --model-type neurosnake_ca
║                                                                       ║
║  # Train with all SOTA features                                       ║
║  python one_click_train_test.py --mode train --model-type neurosnake_ca \\
║      --deduplicate --epochs 100 --visualize                          ║
║                                                                       ║
║  # Quick baseline training                                            ║
║  python one_click_train_test.py --mode train --model-type baseline    ║
║                                                                       ║
║  # Evaluate trained model                                             ║
║  python one_click_train_test.py --mode evaluate                       ║
║                                                                       ║
║  # Full pipeline (train + evaluate)                                   ║
║  python one_click_train_test.py --mode full --visualize               ║
║                                                                       ║
║  # Single image prediction                                            ║
║  python one_click_train_test.py --mode predict --image scan.jpg       ║
╚══════════════════════════════════════════════════════════════════════╝
        """
    )

    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'evaluate', 'predict', 'full', 'info'],
                       help='Operation mode')
    parser.add_argument('--model-type', type=str, default='neurosnake_ca',
                       choices=list(MODEL_TYPES.keys()),
                       help='Model architecture (default: neurosnake_ca)')
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
    parser.add_argument('--deduplicate', action='store_true',
                       help='Run pHash deduplication (prevents data leakage)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--use-baseline-training', action='store_true',
                       help='Use basic training (no Adan/Focal Loss)')

    args = parser.parse_args()

    print_header()
    
    if args.mode == 'info':
        print_info()
        return

    print(f"\n{'Configuration':=^60}")
    print(f"  Mode: {args.mode.upper()}")
    print(f"  Model: {MODEL_TYPES.get(args.model_type, args.model_type)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Deduplication: {'Yes' if args.deduplicate else 'No'}")
    print("=" * 60)

    if args.mode == 'train' or args.mode == 'full':
        train_mode(args)

    if args.mode == 'evaluate' or args.mode == 'full':
        evaluate_mode(args)

    if args.mode == 'predict':
        predict_mode(args)

    print_footer()


def print_header():
    """Print Phoenix Protocol header."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██████╗ ██╗  ██╗ ██████╗ ███████╗███╗   ██╗██╗██╗  ██╗            ║
║   ██╔══██╗██║  ██║██╔═══██╗██╔════╝████╗  ██║██║╚██╗██╔╝            ║
║   ██████╔╝███████║██║   ██║█████╗  ██╔██╗ ██║██║ ╚███╔╝             ║
║   ██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║██║ ██╔██╗             ║
║   ██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║██║██╔╝ ██╗            ║
║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝            ║
║                                                                      ║
║              BRAIN TUMOR DETECTION - SOTA PIPELINE                   ║
║                                                                      ║
║   Features:                                                          ║
║   • NeuroSnake Architecture (Dynamic Snake Convolutions)             ║
║   • Coordinate Attention (Position-Preserving)                       ║
║   • Physics-Informed MRI Augmentation                                ║
║   • Adan Optimizer + Focal Loss                                      ║
║   • pHash Data Deduplication                                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def print_footer():
    """Print completion footer."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    PHOENIX PROTOCOL - COMPLETED                       ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def print_info():
    """Print model information."""
    print("\n" + "="*60)
    print("AVAILABLE MODEL ARCHITECTURES")
    print("="*60)
    for model_type, description in MODEL_TYPES.items():
        marker = " [RECOMMENDED]" if model_type == 'neurosnake_ca' else ""
        print(f"\n  {model_type}:{marker}")
        print(f"    {description}")
    
    print("\n" + "="*60)
    print("SOTA FEATURES (Phoenix Protocol)")
    print("="*60)
    print("""
  1. Dynamic Snake Convolutions
     - Adaptive kernel deformation for irregular tumor boundaries
     - Traces curvilinear features (Glioblastoma infiltrations)
  
  2. Coordinate Attention
     - Preserves spatial position information
     - Critical for tumor location diagnosis
  
  3. MobileViT Block
     - Global context capture
     - Security-hardened against Med-Hammer attacks
  
  4. Physics-Informed Augmentation
     - Elastic deformation (tissue movement)
     - Rician noise (MRI acquisition noise)
     - Intensity inhomogeneity (RF coil bias)
  
  5. Adan Optimizer
     - 1st, 2nd, 3rd moment estimation
     - Superior stability on medical imaging data
  
  6. Focal Loss
     - Handles class imbalance
     - Focuses on hard examples
  
  7. pHash Deduplication
     - Prevents data leakage between splits
     - Honest accuracy evaluation
    """)


def train_mode(args):
    """Run training with Phoenix Protocol."""
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)

    # Check if using SOTA models
    if args.model_type in ['neurosnake', 'neurosnake_ca'] and not args.use_baseline_training:
        # Use Phoenix Protocol training
        from src.train_phoenix import PhoenixProtocolTrainer
        
        trainer = PhoenixProtocolTrainer(
            model_type=args.model_type,
            use_focal_loss=True,
            use_adan_optimizer=True,
            use_physics_augmentation=True
        )
        
        # Run deduplication if requested
        if args.deduplicate:
            try:
                from src.data_deduplication import deduplicate_dataset
                print("\nRunning pHash deduplication...")
                deduplicate_dataset(
                    data_dir=config.DATA_DIR,
                    hamming_threshold=5,
                    output_report=os.path.join(config.RESULTS_DIR, 'deduplication_report.json'),
                    dry_run=True
                )
            except ImportError:
                print("Warning: imagehash not available. Skipping deduplication.")
        
        # Train
        history = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        if args.visualize:
            trainer.save_visualizations()
            
    else:
        # Use baseline training
        from src.train import train_model
        from src.visualize import plot_training_history

        model, history = train_model(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            augment_train=True
        )

        if args.visualize:
            plot_training_history(
                history,
                save_path=os.path.join(config.RESULTS_DIR, 'training_history.png'),
                show=False
            )

    print(f"\n✓ Training completed!")
    print(f"  Best model saved to: {config.BEST_MODEL_PATH}")


def evaluate_mode(args):
    """Run evaluation."""
    print("\n" + "="*60)
    print("EVALUATION MODE")
    print("="*60)

    from src.evaluate import evaluate_from_file
    from src.visualize import visualize_evaluation_results

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first with: python one_click_train_test.py --mode train")
        return

    metrics = evaluate_from_file(args.model_path)

    if args.visualize:
        visualize_evaluation_results(metrics, output_dir=config.RESULTS_DIR, show=False)

    print(f"\n✓ Evaluation completed!")
    print(f"  Results saved to: {config.RESULTS_DIR}")


def predict_mode(args):
    """Run prediction."""
    print("\n" + "="*60)
    print("PREDICTION MODE")
    print("="*60)

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
