"""
Training Module for Brain Tumor Detection.

This module provides functions for training the brain tumor detection model
with support for callbacks, checkpointing, and early stopping.
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import create_data_generators
from models.cnn_model import get_model


def create_callbacks(
    model_save_path: str = config.BEST_MODEL_PATH,
    log_dir: Optional[str] = None
) -> list:
    """
    Create training callbacks for model checkpointing, early stopping, etc.

    Args:
        model_save_path: Path to save the best model
        log_dir: Directory for TensorBoard logs

    Returns:
        List of Keras callbacks
    """
    callbacks = []

    # Model checkpoint - save best model
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
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        mode='min',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr)

    # CSV Logger
    csv_logger = CSVLogger(
        os.path.join(config.RESULTS_DIR, 'training_log.csv'),
        append=False
    )
    callbacks.append(csv_logger)

    # TensorBoard (optional)
    if log_dir:
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)

    return callbacks


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = config.LEARNING_RATE
) -> tf.keras.Model:
    """
    Compile the model with optimizer, loss function, and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled model
    """
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(
    model_type: str = 'baseline',
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    augment_train: bool = True,
    use_tensorboard: bool = False,
    verbose: int = 1
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train the brain tumor detection model.

    Args:
        model_type: Type of model to train ('baseline', 'simple', 'regularized')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        augment_train: Whether to use data augmentation
        use_tensorboard: Whether to use TensorBoard logging
        verbose: Verbosity level

    Returns:
        Tuple of (trained model, training history dict)
    """
    print("=" * 60)
    print("Brain Tumor Detection - Model Training")
    print("=" * 60)

    # Create model
    print(f"\nCreating {model_type} model...")
    model = get_model(model_type)
    model = compile_model(model, learning_rate)

    if verbose:
        model.summary()

    # Create data generators
    print("\nLoading training data...")
    train_generator, validation_generator = create_data_generators(
        batch_size=batch_size,
        augment_train=augment_train
    )

    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")

    # Create callbacks
    log_dir = os.path.join(config.RESULTS_DIR, 'logs') if use_tensorboard else None
    callbacks = create_callbacks(log_dir=log_dir)

    # Train the model
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=verbose
    )

    # Save final model
    final_model_path = os.path.join(config.MODELS_DIR, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    return model, history.history


def train_with_class_weights(
    model_type: str = 'baseline',
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    Train model with class weights to handle imbalanced data.

    Args:
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer

    Returns:
        Tuple of (trained model, training history dict)
    """
    import numpy as np

    # Create model
    model = get_model(model_type)
    model = compile_model(model, learning_rate)

    # Create data generators
    train_generator, validation_generator = create_data_generators(
        batch_size=batch_size,
        augment_train=True
    )

    # Calculate class weights
    total_samples = train_generator.samples
    class_counts = {}
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
        else:
            class_counts[class_name] = 0

    class_weights = {}
    for idx, class_name in enumerate(config.CLASS_NAMES):
        if class_counts[class_name] > 0:
            class_weights[idx] = total_samples / (len(config.CLASS_NAMES) * class_counts[class_name])
        else:
            class_weights[idx] = 1.0

    print(f"Class weights: {class_weights}")

    # Create callbacks
    callbacks = create_callbacks()

    # Train with class weights
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return model, history.history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train brain tumor detection model')
    parser.add_argument('--model-type', type=str, default='baseline',
                       choices=['baseline', 'simple', 'regularized'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging')

    args = parser.parse_args()

    model, history = train_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        augment_train=not args.no_augment,
        use_tensorboard=args.tensorboard
    )
