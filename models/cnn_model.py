"""
CNN Model for Brain Tumor Detection.

This module defines the baseline CNN architecture for binary classification
of brain MRI images (tumor vs. no tumor).
"""

import os
import sys

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.regularizers import l2

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_baseline_cnn(
    input_shape: tuple = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    dropout_rate: float = 0.5
) -> Sequential:
    """
    Create a baseline CNN model for brain tumor detection.

    Architecture:
        - 4 Convolutional blocks with BatchNorm and MaxPooling
        - Global Average Pooling
        - Dense layers with Dropout
        - Softmax output

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classification head
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    return model


def create_simple_cnn(
    input_shape: tuple = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES
) -> Sequential:
    """
    Create a simpler CNN model for quick experiments.

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes

    Returns:
        Keras Sequential model
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Classification head
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


def create_regularized_cnn(
    input_shape: tuple = config.INPUT_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    l2_reg: float = 0.01,
    dropout_rate: float = 0.5
) -> Sequential:
    """
    Create a CNN model with L2 regularization for better generalization.

    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        l2_reg: L2 regularization factor
        dropout_rate: Dropout rate for regularization

    Returns:
        Keras Sequential model
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(l2_reg), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Classification head
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    return model


def get_model(model_type: str = 'baseline', **kwargs) -> Sequential:
    """
    Factory function to create different model architectures.

    Args:
        model_type: Type of model ('baseline', 'simple', 'regularized')
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Keras model

    Raises:
        ValueError: If model_type is not recognized
    """
    model_creators = {
        'baseline': create_baseline_cnn,
        'simple': create_simple_cnn,
        'regularized': create_regularized_cnn
    }

    if model_type not in model_creators:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {list(model_creators.keys())}")

    return model_creators[model_type](**kwargs)


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved model from disk.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)


def save_model(model: tf.keras.Model, save_path: str) -> None:
    """
    Save a model to disk.

    Args:
        model: Keras model to save
        save_path: Path to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    # Test model creation
    print("Testing CNN model creation...")

    for model_type in ['baseline', 'simple', 'regularized']:
        print(f"\n{model_type.upper()} Model:")
        model = get_model(model_type)
        model.summary()
        print(f"Total parameters: {model.count_params():,}")
