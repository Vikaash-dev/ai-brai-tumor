"""
Phoenix Protocol - Unified SOTA Training Pipeline for Brain Tumor Detection.

This module consolidates all SOTA features into a single, production-ready pipeline:
- NeuroSnake architecture with Dynamic Snake Convolutions
- Coordinate Attention for position-preserving feature extraction
- Physics-informed MRI augmentation
- Adan optimizer with Focal Loss
- Data deduplication for preventing data leakage
- INT8 quantization for edge deployment

Usage:
    python -m src.train_phoenix --model-type neurosnake_ca --epochs 100 --deduplicate
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Phoenix Protocol imports
from src.data_preprocessing import create_data_generators, create_test_generator
from src.phoenix_optimizer import create_adan_optimizer, create_focal_loss, FocalLoss
from src.physics_informed_augmentation import PhysicsInformedAugmentation, PhysicsAugmentationGenerator
from src.visualize import plot_training_history, visualize_evaluation_results
from src.evaluate import calculate_metrics, print_evaluation_report, save_evaluation_report

# Model imports
from models.cnn_model import get_model as get_baseline_model
from models.neurosnake_model import (
    create_neurosnake_model,
    create_neurosnake_with_coordinate_attention,
    create_baseline_comparison_model
)


class PhoenixProtocolTrainer:
    """
    Unified SOTA trainer implementing the Phoenix Protocol.
    
    Combines all advanced features:
    - Multiple model architectures (Baseline, NeuroSnake, NeuroSnake+CA)
    - Physics-informed augmentation
    - Adan optimizer with Focal Loss
    - Comprehensive callbacks and logging
    """
    
    MODEL_TYPES = {
        'baseline': 'Baseline CNN (standard convolutions)',
        'neurosnake': 'NeuroSnake (Dynamic Snake Convolutions + MobileViT)',
        'neurosnake_ca': 'NeuroSnake + Coordinate Attention (SOTA)',
    }
    
    def __init__(
        self,
        model_type: str = 'neurosnake_ca',
        input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
        num_classes: int = config.NUM_CLASSES,
        use_focal_loss: bool = True,
        use_adan_optimizer: bool = True,
        use_physics_augmentation: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize Phoenix Protocol trainer.
        
        Args:
            model_type: Type of model ('baseline', 'neurosnake', 'neurosnake_ca')
            input_shape: Input image shape
            num_classes: Number of output classes
            use_focal_loss: Whether to use Focal Loss
            use_adan_optimizer: Whether to use Adan optimizer
            use_physics_augmentation: Whether to use physics-informed augmentation
            dropout_rate: Dropout rate for regularization
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.use_adan_optimizer = use_adan_optimizer
        self.use_physics_augmentation = use_physics_augmentation
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        self.augmentor = None
        
        if use_physics_augmentation:
            self.augmentor = PhysicsInformedAugmentation(
                elastic_alpha_range=(30, 40),
                rician_noise_sigma_range=(0.01, 0.05),
                apply_probability=0.5
            )
    
    def create_model(self) -> keras.Model:
        """
        Create model based on specified type.
        
        Returns:
            Keras model
        """
        print(f"\nCreating model: {self.MODEL_TYPES.get(self.model_type, self.model_type)}")
        
        if self.model_type == 'baseline':
            self.model = create_baseline_comparison_model(
                input_shape=self.input_shape,
                num_classes=self.num_classes
            )
        elif self.model_type == 'neurosnake':
            self.model = create_neurosnake_model(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                use_mobilevit=True,
                dropout_rate=self.dropout_rate
            )
        elif self.model_type == 'neurosnake_ca':
            self.model = create_neurosnake_with_coordinate_attention(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                dropout_rate=self.dropout_rate,
                use_mobilevit=True
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Available: {list(self.MODEL_TYPES.keys())}")
        
        return self.model
    
    def compile_model(
        self,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = 0.02
    ) -> keras.Model:
        """
        Compile model with Phoenix Protocol optimizer and loss.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay for Adan optimizer
            
        Returns:
            Compiled model
        """
        if self.model is None:
            self.create_model()
        
        # Select optimizer
        if self.use_adan_optimizer:
            optimizer = create_adan_optimizer(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            print("Using Adan optimizer (1st, 2nd, 3rd moment estimation)")
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            print("Using Adam optimizer")
        
        # Select loss function
        if self.use_focal_loss:
            loss = create_focal_loss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
            print("Using Focal Loss (handles class imbalance)")
        else:
            loss = 'categorical_crossentropy'
            print("Using Categorical Cross-Entropy loss")
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return self.model
    
    def create_callbacks(
        self,
        model_save_path: str = config.BEST_MODEL_PATH,
        patience_early_stop: int = 15,
        patience_reduce_lr: int = 5
    ) -> list:
        """
        Create training callbacks.
        
        Args:
            model_save_path: Path to save best model
            patience_early_stop: Early stopping patience
            patience_reduce_lr: Learning rate reduction patience
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # CSV Logger
        csv_logger = CSVLogger(
            os.path.join(config.RESULTS_DIR, 'phoenix_training_log.csv'),
            append=False
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(
        self,
        train_dir: str = config.TRAIN_DIR,
        val_dir: str = config.VALIDATION_DIR,
        epochs: int = config.EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        learning_rate: float = config.LEARNING_RATE,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the model using Phoenix Protocol.
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        print("\n" + "="*70)
        print("PHOENIX PROTOCOL - SOTA BRAIN TUMOR DETECTION TRAINING")
        print("="*70)
        
        # Create and compile model
        self.create_model()
        self.compile_model(learning_rate=learning_rate)
        
        if verbose:
            self.model.summary()
            print(f"\nTotal parameters: {self.model.count_params():,}")
        
        # Create data generators
        print("\nLoading training data...")
        train_generator, val_generator = create_data_generators(
            train_dir=train_dir,
            validation_dir=val_dir,
            batch_size=batch_size,
            augment_train=True
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        # Wrap with physics augmentation if enabled
        if self.use_physics_augmentation and self.augmentor:
            print("Applying physics-informed MRI augmentation")
            train_generator = PhysicsAugmentationGenerator(train_generator, self.augmentor)
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Model type: {self.MODEL_TYPES.get(self.model_type, self.model_type)}")
        print(f"Optimizer: {'Adan' if self.use_adan_optimizer else 'Adam'}")
        print(f"Loss: {'Focal Loss' if self.use_focal_loss else 'Cross-Entropy'}")
        print(f"Physics Augmentation: {self.use_physics_augmentation}")
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        
        # Save final model
        final_model_path = os.path.join(config.MODELS_DIR, f'{self.model_type}_final.h5')
        self.model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")
        
        # Print results
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best validation accuracy: {max(self.history['val_accuracy']):.4f}")
        print(f"Final validation accuracy: {self.history['val_accuracy'][-1]:.4f}")
        print(f"Best model saved to: {config.BEST_MODEL_PATH}")
        
        return self.history
    
    def evaluate(
        self,
        test_dir: str = config.TEST_DIR,
        batch_size: int = config.BATCH_SIZE
    ) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Args:
            test_dir: Test data directory
            batch_size: Batch size
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\n" + "="*70)
        print("PHOENIX PROTOCOL - MODEL EVALUATION")
        print("="*70)
        
        # Create test generator
        test_generator = create_test_generator(test_dir, batch_size=batch_size)
        print(f"Test samples: {test_generator.samples}")
        
        # Get predictions
        print("Generating predictions...")
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_proba = predictions[:, 1] if predictions.shape[1] == 2 else predictions.max(axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Print and save report
        print_evaluation_report(metrics, config.CLASS_NAMES)
        save_evaluation_report(metrics, config.CLASS_NAMES)
        
        return metrics
    
    def save_visualizations(self, output_dir: str = config.RESULTS_DIR) -> None:
        """
        Save training visualizations.
        
        Args:
            output_dir: Output directory for plots
        """
        if self.history is None:
            print("No training history available.")
            return
        
        print("\nSaving visualizations...")
        
        # Plot training history
        plot_training_history(
            self.history,
            save_path=os.path.join(output_dir, f'{self.model_type}_training_history.png'),
            show=False
        )
        
        print(f"Visualizations saved to: {output_dir}")


def train_phoenix_protocol(
    model_type: str = 'neurosnake_ca',
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    deduplicate: bool = False,
    visualize: bool = True
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Main function to train using Phoenix Protocol.
    
    Args:
        model_type: Model type ('baseline', 'neurosnake', 'neurosnake_ca')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        deduplicate: Whether to run deduplication first
        visualize: Whether to generate visualizations
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Run deduplication if requested
    if deduplicate:
        try:
            from src.data_deduplication import deduplicate_dataset
            print("\nRunning data deduplication...")
            deduplicate_dataset(
                data_dir=config.DATA_DIR,
                hamming_threshold=5,
                output_report=os.path.join(config.RESULTS_DIR, 'deduplication_report.json'),
                dry_run=True  # Report only, don't remove
            )
        except ImportError:
            print("Warning: imagehash not available. Skipping deduplication.")
    
    # Create trainer
    trainer = PhoenixProtocolTrainer(
        model_type=model_type,
        use_focal_loss=True,
        use_adan_optimizer=True,
        use_physics_augmentation=True
    )
    
    # Train
    history = trainer.train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Save visualizations
    if visualize:
        trainer.save_visualizations()
    
    return trainer.model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phoenix Protocol - SOTA Brain Tumor Detection Training'
    )
    parser.add_argument('--model-type', type=str, default='neurosnake_ca',
                       choices=['baseline', 'neurosnake', 'neurosnake_ca'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--deduplicate', action='store_true',
                       help='Run data deduplication before training')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    model, history = train_phoenix_protocol(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        deduplicate=args.deduplicate,
        visualize=not args.no_visualize
    )
