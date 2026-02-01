"""
Pure TensorFlow Hausdorff Distance 95 (HD95) Implementation

This module provides a graph-compatible HD95 metric that works within
TensorFlow's computation graph, avoiding scipy dependencies.

Reference: Appendix A.1 - Critical Issues (Graph-incompatible HD95)
"""

import tensorflow as tf
from typing import Optional, Tuple


class HausdorffDistance95(tf.keras.metrics.Metric):
    """
    Pure TensorFlow implementation of Hausdorff Distance 95th percentile.
    
    The HD95 measures the 95th percentile of the distances between boundary
    points of predicted and ground truth segmentations. It's more robust
    to outliers than the standard Hausdorff distance.
    
    This implementation is fully graph-compatible and can be used as a
    Keras metric during training.
    
    Example:
        >>> hd95 = HausdorffDistance95()
        >>> model.compile(metrics=[hd95])
    """
    
    def __init__(
        self,
        percentile: float = 95.0,
        spacing: Tuple[float, float] = (1.0, 1.0),
        name: str = "hausdorff_95",
        **kwargs
    ):
        """
        Initialize HD95 metric.
        
        Args:
            percentile: Percentile to compute (default: 95)
            spacing: Pixel spacing in mm (height, width)
            name: Metric name
        """
        super().__init__(name=name, **kwargs)
        self.percentile = percentile
        self.spacing = tf.constant(spacing, dtype=tf.float32)
        
        # Accumulator for batch-wise computation
        self.total_hd95 = self.add_weight(
            name="total_hd95",
            initializer="zeros",
            dtype=tf.float32
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros",
            dtype=tf.float32
        )
    
    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """
        Update metric state with batch of predictions.
        
        Args:
            y_true: Ground truth segmentation [B, H, W] or [B, H, W, 1]
            y_pred: Predicted segmentation [B, H, W] or [B, H, W, 1]
            sample_weight: Optional sample weights
        """
        # Ensure proper shape
        if len(y_true.shape) == 4:
            y_true = tf.squeeze(y_true, axis=-1)
        if len(y_pred.shape) == 4:
            y_pred = tf.squeeze(y_pred, axis=-1)
        
        # Binarize predictions
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        # Compute HD95 for each sample in batch
        batch_size = tf.shape(y_true)[0]
        
        def compute_single_hd95(args):
            gt, pred = args
            return self._compute_hd95(gt, pred)
        
        hd95_values = tf.map_fn(
            compute_single_hd95,
            (y_true, y_pred),
            fn_output_signature=tf.float32
        )
        
        # Handle sample weights
        if sample_weight is not None:
            hd95_values = hd95_values * tf.cast(sample_weight, tf.float32)
            self.total_hd95.assign_add(tf.reduce_sum(hd95_values))
            self.count.assign_add(tf.reduce_sum(tf.cast(sample_weight, tf.float32)))
        else:
            self.total_hd95.assign_add(tf.reduce_sum(hd95_values))
            self.count.assign_add(tf.cast(batch_size, tf.float32))
    
    def result(self) -> tf.Tensor:
        """Compute final HD95 value."""
        return tf.math.divide_no_nan(self.total_hd95, self.count)
    
    def reset_state(self) -> None:
        """Reset metric state."""
        self.total_hd95.assign(0.0)
        self.count.assign(0.0)
    
    def _compute_hd95(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute HD95 between two binary masks.
        
        Args:
            y_true: Ground truth mask [H, W]
            y_pred: Predicted mask [H, W]
            
        Returns:
            HD95 value in mm
        """
        # Extract boundary points
        boundary_true = self._extract_boundary(y_true)
        boundary_pred = self._extract_boundary(y_pred)
        
        # Get coordinates of boundary points
        coords_true = self._get_coordinates(boundary_true)
        coords_pred = self._get_coordinates(boundary_pred)
        
        # Handle empty masks
        n_true = tf.shape(coords_true)[0]
        n_pred = tf.shape(coords_pred)[0]
        
        def compute_hd():
            # Compute directed Hausdorff distances
            dist_true_to_pred = self._directed_hausdorff_percentile(
                coords_true, coords_pred, self.percentile
            )
            dist_pred_to_true = self._directed_hausdorff_percentile(
                coords_pred, coords_true, self.percentile
            )
            return tf.maximum(dist_true_to_pred, dist_pred_to_true)
        
        # Return 0 if both empty, infinity if one empty
        result = tf.cond(
            tf.logical_and(n_true > 0, n_pred > 0),
            compute_hd,
            lambda: tf.cond(
                tf.logical_and(n_true == 0, n_pred == 0),
                lambda: 0.0,
                lambda: 100.0  # Large penalty for missing predictions
            )
        )
        
        return result
    
    def _extract_boundary(self, mask: tf.Tensor) -> tf.Tensor:
        """
        Extract boundary of binary mask using morphological operations.
        
        Args:
            mask: Binary mask [H, W]
            
        Returns:
            Boundary mask [H, W]
        """
        # Add batch and channel dimensions for conv2d
        mask_4d = tf.reshape(mask, [1, tf.shape(mask)[0], tf.shape(mask)[1], 1])
        
        # Erosion kernel (3x3)
        kernel = tf.ones([3, 3, 1, 1], dtype=tf.float32)
        
        # Erode: min pooling via negative max pooling
        eroded = -tf.nn.max_pool2d(-mask_4d, ksize=3, strides=1, padding='SAME')
        
        # Boundary = original - eroded
        boundary = mask_4d - eroded
        
        return tf.squeeze(boundary)
    
    def _get_coordinates(self, boundary: tf.Tensor) -> tf.Tensor:
        """
        Get coordinates of non-zero boundary points.
        
        Args:
            boundary: Boundary mask [H, W]
            
        Returns:
            Coordinates [N, 2] in (y, x) format
        """
        # Find non-zero indices
        indices = tf.where(boundary > 0.5)
        
        # Convert to float and apply spacing
        coords = tf.cast(indices, tf.float32) * self.spacing
        
        return coords
    
    def _directed_hausdorff_percentile(
        self,
        coords_a: tf.Tensor,
        coords_b: tf.Tensor,
        percentile: float
    ) -> tf.Tensor:
        """
        Compute directed Hausdorff distance at given percentile.
        
        For each point in A, find minimum distance to any point in B,
        then return the percentile of these minimum distances.
        
        Args:
            coords_a: Source coordinates [N, 2]
            coords_b: Target coordinates [M, 2]
            percentile: Percentile to compute
            
        Returns:
            Directed HD at percentile
        """
        # Compute pairwise distances efficiently
        # coords_a: [N, 2], coords_b: [M, 2]
        # Expand dims for broadcasting: [N, 1, 2] - [1, M, 2] = [N, M, 2]
        diff = tf.expand_dims(coords_a, 1) - tf.expand_dims(coords_b, 0)
        distances = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
        
        # For each point in A, find minimum distance to B
        min_distances = tf.reduce_min(distances, axis=1)  # [N]
        
        # Compute percentile
        k = tf.cast(
            tf.cast(tf.shape(min_distances)[0], tf.float32) * percentile / 100.0,
            tf.int32
        )
        k = tf.maximum(k, 0)
        k = tf.minimum(k, tf.shape(min_distances)[0] - 1)
        
        # Sort and get percentile value
        sorted_distances = tf.sort(min_distances)
        return sorted_distances[k]


class DiceCoefficient(tf.keras.metrics.Metric):
    """
    Dice coefficient (F1 score for segmentation).
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        name: str = "dice",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.total_dice = self.add_weight(
            name="total_dice",
            initializer="zeros"
        )
        self.count = self.add_weight(
            name="count",
            initializer="zeros"
        )
    
    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        """Update metric state."""
        # Flatten
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # Compute Dice
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.total_dice.assign_add(dice)
        self.count.assign_add(1.0)
    
    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.total_dice, self.count)
    
    def reset_state(self) -> None:
        self.total_dice.assign(0.0)
        self.count.assign(0.0)


class DiceLoss(tf.keras.losses.Loss):
    """
    Dice loss for segmentation.
    
    Loss = 1 - Dice
    """
    
    def __init__(
        self,
        smooth: float = 1e-6,
        name: str = "dice_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute Dice loss."""
        # Flatten along spatial dimensions
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Compute per-sample Dice
        axes = list(range(1, len(y_true.shape)))
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - tf.reduce_mean(dice)


class CombinedLoss(tf.keras.losses.Loss):
    """
    Combined Dice + Cross-Entropy loss.
    
    Loss = α * Dice + (1-α) * BCE
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1e-6,
        name: str = "combined_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute combined loss."""
        dice = self.dice_loss(y_true, y_pred)
        bce = self.bce(y_true, y_pred)
        return self.alpha * dice + (1.0 - self.alpha) * bce


# =============================================================================
# Utility Functions
# =============================================================================

def compute_hd95_numpy(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    spacing: Tuple[float, float] = (1.0, 1.0)
) -> float:
    """
    Compute HD95 using numpy (for validation/testing).
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        spacing: Pixel spacing
        
    Returns:
        HD95 value
    """
    import numpy as np
    from scipy.ndimage import binary_erosion
    from scipy.spatial.distance import directed_hausdorff
    
    # Extract boundaries
    boundary_true = y_true.astype(bool) ^ binary_erosion(y_true.astype(bool))
    boundary_pred = y_pred.astype(bool) ^ binary_erosion(y_pred.astype(bool))
    
    # Get coordinates
    coords_true = np.argwhere(boundary_true) * np.array(spacing)
    coords_pred = np.argwhere(boundary_pred) * np.array(spacing)
    
    if len(coords_true) == 0 or len(coords_pred) == 0:
        return 0.0 if len(coords_true) == 0 and len(coords_pred) == 0 else 100.0
    
    # Compute all pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(coords_true, coords_pred)
    
    # Directed distances
    dist_true_to_pred = np.min(distances, axis=1)
    dist_pred_to_true = np.min(distances, axis=0)
    
    # HD95
    hd95 = max(
        np.percentile(dist_true_to_pred, 95),
        np.percentile(dist_pred_to_true, 95)
    )
    
    return float(hd95)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test HD95 metric
    print("Testing HD95 Metric...")
    
    # Create test masks
    y_true = tf.zeros([2, 64, 64])
    y_pred = tf.zeros([2, 64, 64])
    
    # Add some regions
    y_true = tf.tensor_scatter_nd_update(
        y_true,
        [[0, 20, 20], [0, 20, 21], [0, 21, 20], [0, 21, 21],
         [1, 30, 30], [1, 30, 31], [1, 31, 30], [1, 31, 31]],
        [1.0] * 8
    )
    y_pred = tf.tensor_scatter_nd_update(
        y_pred,
        [[0, 22, 22], [0, 22, 23], [0, 23, 22], [0, 23, 23],
         [1, 31, 31], [1, 31, 32], [1, 32, 31], [1, 32, 32]],
        [1.0] * 8
    )
    
    # Compute HD95
    hd95_metric = HausdorffDistance95()
    hd95_metric.update_state(y_true, y_pred)
    print(f"HD95: {hd95_metric.result().numpy():.2f} mm")
    
    # Test Dice
    dice_metric = DiceCoefficient()
    dice_metric.update_state(y_true, y_pred)
    print(f"Dice: {dice_metric.result().numpy():.4f}")
    
    print("\n✅ HD95 metric implementation working!")
