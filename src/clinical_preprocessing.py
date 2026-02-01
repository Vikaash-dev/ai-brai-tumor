# Clinical Preprocessing Pipeline for Medical MRI Images
"""
Implements clinical-grade preprocessing from SOTA medical imaging pipelines:
- nnU-Net preprocessing techniques
- MONAI transforms
- Standard neuroradiology workflows

Features:
- Skull stripping (brain extraction)
- N4 bias field correction
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Z-score normalization per scan
- Resolution standardization
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for clinical preprocessing."""
    target_size: Tuple[int, int] = (224, 224)
    apply_skull_strip: bool = True
    apply_bias_correction: bool = True
    apply_clahe: bool = True
    apply_normalization: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'percentile'


class SkullStripper:
    """
    Simple skull stripping using morphological operations.
    For production, use FSL BET or ANTs brain extraction.
    """
    
    def __init__(
        self,
        threshold_method: str = 'otsu',
        kernel_size: int = 5,
        iterations: int = 3
    ):
        self.threshold_method = threshold_method
        self.kernel_size = kernel_size
        self.iterations = iterations
        
    def strip(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract brain region from MRI image.
        
        Args:
            image: Input MRI image (H, W) or (H, W, C)
            
        Returns:
            Tuple of (brain_extracted_image, brain_mask)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                (image * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            )
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Apply threshold
        if self.threshold_method == 'otsu':
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.kernel_size, self.kernel_size)
        )
        
        # Close holes
        mask = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            kernel, iterations=self.iterations
        )
        
        # Remove small objects
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            kernel, iterations=1
        )
        
        # Find largest connected component (brain)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255
        
        # Apply mask to image
        mask_normalized = mask / 255.0
        
        if len(image.shape) == 3:
            brain_extracted = image * mask_normalized[:, :, np.newaxis]
        else:
            brain_extracted = image * mask_normalized
            
        return brain_extracted, mask_normalized


class BiasFieldCorrector:
    """
    N4 Bias Field Correction approximation.
    For full N4, use ANTs N4BiasFieldCorrection.
    """
    
    def __init__(
        self,
        downsample_factor: int = 4,
        smoothing_sigma: float = 50.0,
        iterations: int = 3
    ):
        self.downsample_factor = downsample_factor
        self.smoothing_sigma = smoothing_sigma
        self.iterations = iterations
        
    def correct(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bias field correction.
        
        Args:
            image: Input MRI image
            
        Returns:
            Bias-corrected image
        """
        # Convert to grayscale for bias estimation
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()
        
        # Avoid division by zero
        epsilon = 1e-7
        gray = np.clip(gray, epsilon, 1.0)
        
        # Log transform
        log_image = np.log(gray + epsilon)
        
        # Estimate bias field using low-pass filtering
        for _ in range(self.iterations):
            # Downsample
            h, w = log_image.shape
            small = cv2.resize(
                log_image,
                (w // self.downsample_factor, h // self.downsample_factor)
            )
            
            # Smooth
            bias_small = cv2.GaussianBlur(
                small,
                (0, 0),
                self.smoothing_sigma / self.downsample_factor
            )
            
            # Upsample
            bias_field = cv2.resize(bias_small, (w, h))
            
            # Subtract bias from log image
            log_image = log_image - bias_field
        
        # Exp transform
        corrected = np.exp(log_image) - epsilon
        
        # Normalize to [0, 1]
        corrected = (corrected - corrected.min()) / (corrected.max() - corrected.min() + epsilon)
        
        # Apply to original image
        if len(image.shape) == 3:
            # Apply correction to each channel
            correction_factor = corrected / (gray + epsilon)
            corrected_image = image * correction_factor[:, :, np.newaxis]
        else:
            corrected_image = corrected
            
        return np.clip(corrected_image, 0, 1)


class CLAHEEnhancer:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Improves local contrast while limiting noise amplification.
    """
    
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement.
        
        Args:
            image: Input image [0, 1]
            
        Returns:
            Enhanced image [0, 1]
        """
        # Convert to uint8
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            image_uint8 = (image * 255).astype(np.uint8)
            lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size
            )
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced / 255.0
        else:
            image_uint8 = (image * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid_size
            )
            enhanced = clahe.apply(image_uint8)
            return enhanced / 255.0


class IntensityNormalizer:
    """
    Intensity normalization methods for MRI images.
    """
    
    def __init__(self, method: str = 'zscore'):
        """
        Initialize normalizer.
        
        Args:
            method: 'zscore', 'minmax', or 'percentile'
        """
        self.method = method
        
    def normalize(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Normalize image intensity.
        
        Args:
            image: Input image
            mask: Optional brain mask for computing statistics
            
        Returns:
            Normalized image
        """
        if mask is not None:
            # Compute stats only on brain region
            brain_pixels = image[mask > 0.5]
        else:
            brain_pixels = image.flatten()
        
        if self.method == 'zscore':
            mean = np.mean(brain_pixels)
            std = np.std(brain_pixels)
            if std < 1e-8:
                std = 1.0
            normalized = (image - mean) / std
            # Clip to reasonable range
            normalized = np.clip(normalized, -5, 5)
            # Scale to [0, 1]
            normalized = (normalized + 5) / 10
            
        elif self.method == 'minmax':
            min_val = np.min(brain_pixels)
            max_val = np.max(brain_pixels)
            if max_val - min_val < 1e-8:
                max_val = min_val + 1.0
            normalized = (image - min_val) / (max_val - min_val)
            
        elif self.method == 'percentile':
            p1 = np.percentile(brain_pixels, 1)
            p99 = np.percentile(brain_pixels, 99)
            if p99 - p1 < 1e-8:
                p99 = p1 + 1.0
            normalized = np.clip(image, p1, p99)
            normalized = (normalized - p1) / (p99 - p1)
            
        else:
            normalized = image
            
        return np.clip(normalized, 0, 1)


class ClinicalPreprocessor:
    """
    Complete clinical preprocessing pipeline.
    Combines all preprocessing steps in correct order.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.skull_stripper = SkullStripper()
        self.bias_corrector = BiasFieldCorrector()
        self.clahe_enhancer = CLAHEEnhancer(
            clip_limit=self.config.clahe_clip_limit,
            tile_grid_size=self.config.clahe_grid_size
        )
        self.normalizer = IntensityNormalizer(
            method=self.config.normalization_method
        )
        
    def preprocess(
        self,
        image: np.ndarray,
        return_intermediates: bool = False
    ) -> Any:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input MRI image (H, W, C) or (H, W)
            return_intermediates: If True, return intermediate results
            
        Returns:
            Preprocessed image, or tuple of (preprocessed, intermediates)
        """
        intermediates = {'original': image.copy()}
        
        # Ensure float [0, 1]
        if image.dtype == np.uint8:
            image = image / 255.0
        elif image.max() > 1.0:
            image = image / 255.0
            
        # Resize to target size
        if image.shape[:2] != self.config.target_size:
            image = cv2.resize(image, self.config.target_size)
        intermediates['resized'] = image.copy()
        
        # Initialize mask as full image
        mask = np.ones(image.shape[:2])
        
        # Step 1: Skull stripping
        if self.config.apply_skull_strip:
            image, mask = self.skull_stripper.strip(image)
            intermediates['skull_stripped'] = image.copy()
            intermediates['brain_mask'] = mask.copy()
        
        # Step 2: Bias field correction
        if self.config.apply_bias_correction:
            image = self.bias_corrector.correct(image)
            intermediates['bias_corrected'] = image.copy()
        
        # Step 3: CLAHE enhancement
        if self.config.apply_clahe:
            image = self.clahe_enhancer.enhance(image)
            intermediates['clahe_enhanced'] = image.copy()
        
        # Step 4: Intensity normalization
        if self.config.apply_normalization:
            image = self.normalizer.normalize(image, mask)
            intermediates['normalized'] = image.copy()
        
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        if return_intermediates:
            return image, intermediates
        return image
    
    def preprocess_batch(
        self,
        images: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: Batch of images (N, H, W, C) or (N, H, W)
            verbose: Print progress
            
        Returns:
            Preprocessed batch
        """
        preprocessed = []
        n_images = len(images)
        
        for i, image in enumerate(images):
            if verbose and (i + 1) % 100 == 0:
                logger.info(f"Preprocessing image {i + 1}/{n_images}")
            preprocessed.append(self.preprocess(image))
        
        return np.array(preprocessed)


def create_preprocessing_pipeline(
    target_size: Tuple[int, int] = (224, 224),
    apply_skull_strip: bool = True,
    apply_bias_correction: bool = True,
    apply_clahe: bool = True,
    normalization: str = 'zscore'
) -> ClinicalPreprocessor:
    """
    Create a clinical preprocessing pipeline.
    
    Args:
        target_size: Output image size
        apply_skull_strip: Whether to apply skull stripping
        apply_bias_correction: Whether to apply bias field correction
        apply_clahe: Whether to apply CLAHE
        normalization: Normalization method
        
    Returns:
        ClinicalPreprocessor instance
    """
    config = PreprocessingConfig(
        target_size=target_size,
        apply_skull_strip=apply_skull_strip,
        apply_bias_correction=apply_bias_correction,
        apply_clahe=apply_clahe,
        apply_normalization=True,
        normalization_method=normalization
    )
    
    return ClinicalPreprocessor(config)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Clinical Preprocessing Pipeline")
    print("=" * 60)
    print("\nPreprocessing steps:")
    print("1. Skull stripping (brain extraction)")
    print("2. N4 bias field correction")
    print("3. CLAHE contrast enhancement")
    print("4. Z-score normalization")
    print("\nBased on nnU-Net and MONAI best practices")
