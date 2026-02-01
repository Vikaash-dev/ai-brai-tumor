"""
Input Validation Module for PHOENIX-v3.1

This module provides comprehensive input validation to prevent:
- OOM (Out-of-Memory) attacks via oversized inputs
- DoS (Denial-of-Service) via malformed data
- NaN/Inf propagation causing silent failures
- Type mismatches causing runtime errors

Reference: Appendix A.2 - Medium Issues (Input Validation)
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    
    # Dimension limits
    max_image_width: int = 512
    max_image_height: int = 512
    max_batch_size: int = 64
    max_channels: int = 16
    
    # Value validation
    check_nan: bool = True
    check_inf: bool = True
    check_range: bool = True
    value_min: float = -10.0  # Generous range for normalized inputs
    value_max: float = 10.0
    
    # Type validation
    expected_dtype: str = "float32"
    allowed_dtypes: Tuple[str, ...] = ("float32", "float16", "float64")
    
    # Memory limits
    max_tensor_size_mb: float = 100.0
    max_total_memory_mb: float = 500.0
    
    # Strict mode (raise exceptions vs warnings)
    strict_mode: bool = True


class InputValidationError(Exception):
    """Custom exception for input validation failures."""
    pass


class InputValidator:
    """
    Comprehensive input validator for medical imaging deep learning.
    
    Features:
    - Dimension validation (prevent OOM)
    - NaN/Inf detection
    - Type checking
    - Memory estimation
    - Batch validation
    
    Example:
        >>> validator = InputValidator()
        >>> image = np.random.randn(224, 224, 3).astype(np.float32)
        >>> validated = validator.validate_image(image)
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the input validator.
        
        Args:
            config: Validation configuration. Uses defaults if None.
        """
        self.config = config or ValidationConfig()
        self._validation_stats = {
            "total_validated": 0,
            "nan_detected": 0,
            "inf_detected": 0,
            "dimension_violations": 0,
            "type_violations": 0,
            "memory_violations": 0
        }
    
    def validate_image(
        self,
        image: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        name: str = "image"
    ) -> np.ndarray:
        """
        Validate a single image array.
        
        Args:
            image: Input image as numpy array
            expected_shape: Expected shape (optional)
            name: Name for error messages
            
        Returns:
            Validated image (possibly corrected dtype)
            
        Raises:
            InputValidationError: If validation fails in strict mode
        """
        self._validation_stats["total_validated"] += 1
        
        # Check if input is array-like
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except Exception as e:
                self._raise_or_warn(f"{name}: Cannot convert to numpy array: {e}")
                return None
        
        # Validate dimensions
        self._validate_dimensions(image, name)
        
        # Validate dtype
        image = self._validate_dtype(image, name)
        
        # Validate values (NaN, Inf, range)
        self._validate_values(image, name)
        
        # Validate memory
        self._validate_memory(image, name)
        
        # Validate expected shape if provided
        if expected_shape is not None:
            self._validate_shape(image, expected_shape, name)
        
        return image
    
    def validate_batch(
        self,
        batch: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
        name: str = "batch"
    ) -> np.ndarray:
        """
        Validate a batch of images.
        
        Args:
            batch: Batch of images [N, H, W, C]
            expected_shape: Expected shape for each image (H, W, C)
            name: Name for error messages
            
        Returns:
            Validated batch
        """
        # Check batch dimension
        if batch.ndim < 4:
            self._raise_or_warn(f"{name}: Expected 4D batch [N, H, W, C], got {batch.ndim}D")
            return None
        
        batch_size = batch.shape[0]
        if batch_size > self.config.max_batch_size:
            self._raise_or_warn(
                f"{name}: Batch size {batch_size} exceeds maximum {self.config.max_batch_size}"
            )
        
        # Validate each image dimension
        for i in range(min(batch_size, 5)):  # Check first 5 samples
            self.validate_image(batch[i], expected_shape, f"{name}[{i}]")
        
        # Validate total memory
        total_memory_mb = batch.nbytes / (1024 * 1024)
        if total_memory_mb > self.config.max_total_memory_mb:
            self._raise_or_warn(
                f"{name}: Total memory {total_memory_mb:.1f}MB exceeds limit "
                f"{self.config.max_total_memory_mb}MB"
            )
        
        return batch
    
    def validate_tensor(
        self,
        tensor: Any,
        name: str = "tensor"
    ) -> Any:
        """
        Validate a TensorFlow/PyTorch tensor by converting to numpy temporarily.
        
        Args:
            tensor: Input tensor
            name: Name for error messages
            
        Returns:
            Original tensor (validated)
        """
        # Try to get numpy representation
        if hasattr(tensor, 'numpy'):
            # TensorFlow eager tensor or PyTorch tensor
            np_array = tensor.numpy()
        elif hasattr(tensor, 'eval'):
            # TensorFlow graph tensor (requires session)
            logger.warning(f"{name}: Graph tensor validation skipped (use eager mode)")
            return tensor
        else:
            np_array = np.array(tensor)
        
        # Validate as numpy
        self.validate_image(np_array, name=name)
        
        return tensor
    
    def _validate_dimensions(self, image: np.ndarray, name: str) -> None:
        """Validate image dimensions."""
        if image.ndim < 2 or image.ndim > 4:
            self._raise_or_warn(f"{name}: Invalid dimensions {image.ndim}, expected 2-4D")
            return
        
        # Get spatial dimensions
        if image.ndim == 2:
            h, w = image.shape
        elif image.ndim == 3:
            h, w, c = image.shape
            if c > self.config.max_channels:
                self._raise_or_warn(
                    f"{name}: Channels {c} exceed maximum {self.config.max_channels}"
                )
        else:  # 4D
            n, h, w, c = image.shape
            if n > self.config.max_batch_size:
                self._validation_stats["dimension_violations"] += 1
                self._raise_or_warn(
                    f"{name}: Batch size {n} exceeds maximum {self.config.max_batch_size}"
                )
        
        # Check spatial dimensions
        if h > self.config.max_image_height:
            self._validation_stats["dimension_violations"] += 1
            self._raise_or_warn(
                f"{name}: Height {h} exceeds maximum {self.config.max_image_height}"
            )
        
        if w > self.config.max_image_width:
            self._validation_stats["dimension_violations"] += 1
            self._raise_or_warn(
                f"{name}: Width {w} exceeds maximum {self.config.max_image_width}"
            )
    
    def _validate_dtype(self, image: np.ndarray, name: str) -> np.ndarray:
        """Validate and possibly correct dtype."""
        dtype_name = str(image.dtype)
        
        if dtype_name not in self.config.allowed_dtypes:
            self._validation_stats["type_violations"] += 1
            
            # Try to convert
            try:
                image = image.astype(self.config.expected_dtype)
                logger.info(f"{name}: Converted dtype from {dtype_name} to {self.config.expected_dtype}")
            except Exception as e:
                self._raise_or_warn(f"{name}: Cannot convert dtype {dtype_name}: {e}")
        
        return image
    
    def _validate_values(self, image: np.ndarray, name: str) -> None:
        """Validate NaN, Inf, and value ranges."""
        # Check for NaN
        if self.config.check_nan:
            nan_count = np.sum(np.isnan(image))
            if nan_count > 0:
                self._validation_stats["nan_detected"] += 1
                self._raise_or_warn(f"{name}: Contains {nan_count} NaN values")
        
        # Check for Inf
        if self.config.check_inf:
            inf_count = np.sum(np.isinf(image))
            if inf_count > 0:
                self._validation_stats["inf_detected"] += 1
                self._raise_or_warn(f"{name}: Contains {inf_count} Inf values")
        
        # Check value range
        if self.config.check_range:
            min_val = np.min(image)
            max_val = np.max(image)
            
            if min_val < self.config.value_min or max_val > self.config.value_max:
                logger.warning(
                    f"{name}: Values [{min_val:.3f}, {max_val:.3f}] outside expected "
                    f"range [{self.config.value_min}, {self.config.value_max}]"
                )
    
    def _validate_memory(self, image: np.ndarray, name: str) -> None:
        """Validate memory footprint."""
        memory_mb = image.nbytes / (1024 * 1024)
        
        if memory_mb > self.config.max_tensor_size_mb:
            self._validation_stats["memory_violations"] += 1
            self._raise_or_warn(
                f"{name}: Memory {memory_mb:.1f}MB exceeds limit "
                f"{self.config.max_tensor_size_mb}MB"
            )
    
    def _validate_shape(
        self,
        image: np.ndarray,
        expected_shape: Tuple[int, ...],
        name: str
    ) -> None:
        """Validate against expected shape."""
        if image.shape != expected_shape:
            self._raise_or_warn(
                f"{name}: Shape {image.shape} does not match expected {expected_shape}"
            )
    
    def _raise_or_warn(self, message: str) -> None:
        """Raise exception or log warning based on strict mode."""
        if self.config.strict_mode:
            raise InputValidationError(message)
        else:
            logger.warning(message)
    
    def get_stats(self) -> dict:
        """Get validation statistics."""
        return self._validation_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        for key in self._validation_stats:
            self._validation_stats[key] = 0


def validate_model_input(
    x: np.ndarray,
    expected_shape: Tuple[int, ...] = (224, 224, 3),
    strict: bool = True
) -> np.ndarray:
    """
    Convenience function to validate model input.
    
    Args:
        x: Input array
        expected_shape: Expected shape
        strict: Whether to raise exceptions
        
    Returns:
        Validated array
    """
    config = ValidationConfig(strict_mode=strict)
    validator = InputValidator(config)
    
    if x.ndim == 4:
        return validator.validate_batch(x, expected_shape)
    else:
        return validator.validate_image(x, expected_shape)


def sanitize_input(
    x: np.ndarray,
    replace_nan: float = 0.0,
    replace_inf: float = 0.0,
    clip_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Sanitize input by replacing invalid values.
    
    Args:
        x: Input array
        replace_nan: Value to replace NaN with
        replace_inf: Value to replace Inf with
        clip_range: Optional (min, max) to clip values
        
    Returns:
        Sanitized array
    """
    x = x.copy()
    
    # Replace NaN
    x = np.nan_to_num(x, nan=replace_nan, posinf=replace_inf, neginf=-replace_inf)
    
    # Clip range
    if clip_range is not None:
        x = np.clip(x, clip_range[0], clip_range[1])
    
    return x


# TensorFlow-specific validation
try:
    import tensorflow as tf
    
    class TFInputValidator:
        """TensorFlow-compatible input validation layer."""
        
        def __init__(self, config: Optional[ValidationConfig] = None):
            self.config = config or ValidationConfig()
        
        @tf.function
        def validate(self, x: tf.Tensor) -> tf.Tensor:
            """
            Validate tensor within TensorFlow graph.
            
            Args:
                x: Input tensor
                
            Returns:
                Validated tensor (with assertions)
            """
            # Check for NaN
            if self.config.check_nan:
                has_nan = tf.reduce_any(tf.math.is_nan(x))
                tf.debugging.assert_equal(
                    has_nan, False,
                    message="Input contains NaN values"
                )
            
            # Check for Inf
            if self.config.check_inf:
                has_inf = tf.reduce_any(tf.math.is_inf(x))
                tf.debugging.assert_equal(
                    has_inf, False,
                    message="Input contains Inf values"
                )
            
            # Check shape bounds
            shape = tf.shape(x)
            if len(x.shape) >= 2:
                tf.debugging.assert_less_equal(
                    shape[-2], self.config.max_image_height,
                    message=f"Height exceeds maximum {self.config.max_image_height}"
                )
                tf.debugging.assert_less_equal(
                    shape[-3] if len(x.shape) > 2 else shape[-2], 
                    self.config.max_image_width,
                    message=f"Width exceeds maximum {self.config.max_image_width}"
                )
            
            return x
        
        def sanitize(self, x: tf.Tensor) -> tf.Tensor:
            """
            Sanitize tensor by replacing invalid values.
            
            Args:
                x: Input tensor
                
            Returns:
                Sanitized tensor
            """
            # Replace NaN with 0
            x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
            
            # Replace Inf with large finite value
            max_val = tf.constant(1e6, dtype=x.dtype)
            x = tf.where(tf.math.is_inf(x), tf.sign(x) * max_val, x)
            
            return x
    
    def create_validation_layer(config: Optional[ValidationConfig] = None):
        """
        Create a Keras layer for input validation.
        
        Args:
            config: Validation configuration
            
        Returns:
            Keras Lambda layer for validation
        """
        config = config or ValidationConfig()
        validator = TFInputValidator(config)
        
        return tf.keras.layers.Lambda(
            validator.validate,
            name="input_validation"
        )

except ImportError:
    logger.info("TensorFlow not available, skipping TF-specific validation")


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Create validator
    config = ValidationConfig(
        max_image_width=256,
        max_image_height=256,
        strict_mode=False
    )
    validator = InputValidator(config)
    
    # Test with valid input
    print("Testing valid input...")
    valid_image = np.random.randn(224, 224, 3).astype(np.float32)
    result = validator.validate_image(valid_image, name="valid_image")
    print(f"Valid image validated: {result is not None}")
    
    # Test with NaN
    print("\nTesting input with NaN...")
    nan_image = np.random.randn(224, 224, 3).astype(np.float32)
    nan_image[100, 100, 0] = np.nan
    result = validator.validate_image(nan_image, name="nan_image")
    
    # Test with oversized input
    print("\nTesting oversized input...")
    oversized = np.random.randn(512, 512, 3).astype(np.float32)
    result = validator.validate_image(oversized, name="oversized")
    
    # Print stats
    print("\nValidation Statistics:")
    print(validator.get_stats())
    
    # Test sanitization
    print("\nTesting sanitization...")
    dirty_image = np.array([[np.nan, np.inf], [1.0, -np.inf]])
    clean_image = sanitize_input(dirty_image, clip_range=(-1, 1))
    print(f"Dirty: {dirty_image}")
    print(f"Clean: {clean_image}")
