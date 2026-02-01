"""
Prediction Module for Brain Tumor Detection.

This module provides functions for making predictions on new MRI images
using trained brain tumor detection models.
"""

import os
import sys
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

# TensorFlow imports
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_preprocessing import load_and_preprocess_image
from models.cnn_model import load_model


def predict_single_image(
    model: tf.keras.Model,
    image_path: str,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Make a prediction for a single MRI image.

    Args:
        model: Trained Keras model
        image_path: Path to the MRI image
        threshold: Classification threshold

    Returns:
        Dictionary containing prediction results
    """
    # Preprocess image
    img = load_and_preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(img, verbose=0)
    probabilities = prediction[0]

    # Get predicted class
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = config.CLASS_NAMES[predicted_class_idx]
    confidence = probabilities[predicted_class_idx]

    result = {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'probabilities': {
            config.CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(config.CLASS_NAMES))
        },
        'is_tumor': predicted_class == 'tumor'
    }

    return result


def predict_batch(
    model: tf.keras.Model,
    image_paths: List[str],
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Make predictions for multiple MRI images.

    Args:
        model: Trained Keras model
        image_paths: List of paths to MRI images
        threshold: Classification threshold

    Returns:
        List of prediction dictionaries
    """
    results = []

    for image_path in image_paths:
        try:
            result = predict_single_image(model, image_path, threshold)
            results.append(result)
        except Exception as e:
            results.append({
                'image_path': image_path,
                'error': str(e)
            })

    return results


def predict_directory(
    model: tf.keras.Model,
    directory: str,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Make predictions for all images in a directory.

    Args:
        model: Trained Keras model
        directory: Path to directory containing images
        threshold: Classification threshold

    Returns:
        List of prediction dictionaries
    """
    # Find all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_paths = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"Found {len(image_paths)} images in {directory}")

    return predict_batch(model, image_paths, threshold)


def print_prediction(result: Dict[str, Any]) -> None:
    """
    Print a formatted prediction result.

    Args:
        result: Prediction result dictionary
    """
    if 'error' in result:
        print(f"Error processing {result['image_path']}: {result['error']}")
        return

    print(f"\nImage: {os.path.basename(result['image_path'])}")
    print(f"  Prediction: {result['predicted_class'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"    - {class_name}: {prob:.2%}")


def save_predictions(
    results: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Save prediction results to a text file.

    Args:
        results: List of prediction dictionaries
        output_path: Path to save the results
    """
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BRAIN TUMOR DETECTION - BATCH PREDICTIONS\n")
        f.write("=" * 60 + "\n\n")

        tumor_count = sum(1 for r in results if r.get('is_tumor', False))
        no_tumor_count = sum(1 for r in results if not r.get('is_tumor', True) and 'error' not in r)
        error_count = sum(1 for r in results if 'error' in r)

        f.write(f"Total images processed: {len(results)}\n")
        f.write(f"  Tumor detected: {tumor_count}\n")
        f.write(f"  No tumor: {no_tumor_count}\n")
        f.write(f"  Errors: {error_count}\n")
        f.write("\n" + "-" * 60 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"[{i}] {result['image_path']}\n")

            if 'error' in result:
                f.write(f"    ERROR: {result['error']}\n")
            else:
                f.write(f"    Prediction: {result['predicted_class']}\n")
                f.write(f"    Confidence: {result['confidence']:.4f}\n")
                f.write(f"    Probabilities: {result['probabilities']}\n")

            f.write("\n")

    print(f"Predictions saved to: {output_path}")


def interactive_prediction(model: tf.keras.Model) -> None:
    """
    Interactive prediction mode - enter image paths to get predictions.

    Args:
        model: Trained Keras model
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 60)
    print("Enter image paths to get predictions.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            image_path = input("Enter image path: ").strip()

            if image_path.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode.")
                break

            if not os.path.exists(image_path):
                print(f"Error: File not found: {image_path}")
                continue

            result = predict_single_image(model, image_path)
            print_prediction(result)

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Make predictions with brain tumor detection model')
    parser.add_argument('--model-path', type=str, default=config.BEST_MODEL_PATH,
                       help='Path to saved model')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--directory', type=str, help='Path to directory for batch prediction')
    parser.add_argument('--output', type=str, default=os.path.join(config.RESULTS_DIR, 'batch_predictions.txt'),
                       help='Output file for batch predictions')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode')

    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train a model first using: python -m src.train")
        sys.exit(1)

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)

    if args.interactive:
        interactive_prediction(model)
    elif args.image:
        result = predict_single_image(model, args.image)
        print_prediction(result)
    elif args.directory:
        results = predict_directory(model, args.directory)
        for result in results:
            print_prediction(result)
        save_predictions(results, args.output)
    else:
        # Default: interactive mode
        interactive_prediction(model)
