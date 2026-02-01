"""
Data Preprocessing Module for Brain Tumor Detection.

This module provides functions for loading, preprocessing, and augmenting
MRI images for brain tumor detection tasks.
"""

import os
import numpy as np
from typing import Tuple, Optional, Generator

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Image processing
from PIL import Image
import cv2

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_data_generators(
    train_dir: str = config.TRAIN_DIR,
    validation_dir: str = config.VALIDATION_DIR,
    img_size: Tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    augment_train: bool = True
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator,
           tf.keras.preprocessing.image.DirectoryIterator]:
    """
    Create training and validation data generators with optional augmentation.

    Args:
        train_dir: Path to training data directory
        validation_dir: Path to validation data directory
        img_size: Target image size (width, height)
        batch_size: Batch size for data generators
        augment_train: Whether to apply data augmentation to training data

    Returns:
        Tuple of (train_generator, validation_generator)
    """
    # Training data generator with augmentation
    if augment_train:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT_RANGE,
            height_shift_range=config.HEIGHT_SHIFT_RANGE,
            shear_range=config.SHEAR_RANGE,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            fill_mode=config.FILL_MODE
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    # Validation data generator (no augmentation)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False
    )

    return train_generator, validation_generator


def create_test_generator(
    test_dir: str = config.TEST_DIR,
    img_size: Tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE
) -> tf.keras.preprocessing.image.DirectoryIterator:
    """
    Create test data generator (no augmentation).

    Args:
        test_dir: Path to test data directory
        img_size: Target image size (width, height)
        batch_size: Batch size for data generator

    Returns:
        Test data generator
    """
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=config.CLASS_NAMES,
        shuffle=False
    )

    return test_generator


def load_and_preprocess_image(
    image_path: str,
    img_size: Tuple[int, int] = config.IMG_SIZE,
    normalize: bool = True
) -> np.ndarray:
    """
    Load and preprocess a single image for prediction.

    Args:
        image_path: Path to the image file
        img_size: Target image size (width, height)
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        Preprocessed image array with shape (1, height, width, channels)
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize(img_size)

    # Convert to numpy array
    img_array = np.array(img)

    # Normalize if requested
    if normalize:
        img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def load_images_from_directory(
    directory: str,
    img_size: Tuple[int, int] = config.IMG_SIZE,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load all images from a directory structure (class-based folders).

    Args:
        directory: Root directory containing class subdirectories
        img_size: Target image size (width, height)
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        Tuple of (images array, labels array, file paths list)
    """
    images = []
    labels = []
    paths = []

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(class_dir, filename)
                try:
                    img = load_and_preprocess_image(file_path, img_size, normalize)
                    images.append(img[0])  # Remove batch dimension
                    labels.append(class_idx)
                    paths.append(file_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    return np.array(images), np.array(labels), paths


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance MRI images.

    Args:
        image: Input image (grayscale or RGB)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Enhanced image
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)

        # Merge channels back
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)

    return enhanced


def get_dataset_statistics(directory: str) -> dict:
    """
    Get statistics about the dataset.

    Args:
        directory: Root directory containing class subdirectories

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_images': 0,
        'class_distribution': {},
        'directory': directory
    }

    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(directory, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
            stats['class_distribution'][class_name] = count
            stats['total_images'] += count
        else:
            stats['class_distribution'][class_name] = 0

    return stats


if __name__ == '__main__':
    # Test the preprocessing functions
    print("Testing data preprocessing module...")

    # Test dataset statistics
    for split_name, split_dir in [('train', config.TRAIN_DIR),
                                   ('validation', config.VALIDATION_DIR),
                                   ('test', config.TEST_DIR)]:
        stats = get_dataset_statistics(split_dir)
        print(f"\n{split_name.upper()} Dataset Statistics:")
        print(f"  Total images: {stats['total_images']}")
        for class_name, count in stats['class_distribution'].items():
            print(f"  {class_name}: {count}")
