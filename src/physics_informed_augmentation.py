"""
Physics-Informed Augmentation Module for Phoenix Protocol.
Implements MRI-specific augmentation strategies that simulate real-world artifacts.

Unlike generic augmentation, these transforms are based on the physics of MRI acquisition:
- Elastic deformation (tissue deformation)
- Rician noise (MRI noise model)
- Intensity inhomogeneity (RF coil bias field)
- Ghosting artifacts (motion-induced)
"""

import numpy as np
from typing import Tuple, Optional
import cv2

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from scipy.ndimage import map_coordinates, gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PhysicsInformedAugmentation:
    """
    MRI-specific augmentation that simulates real-world imaging artifacts.
    Based on the physics of MRI acquisition for more realistic training data.
    """
    
    def __init__(
        self,
        elastic_alpha_range: Tuple[float, float] = (30, 40),
        elastic_sigma: float = 5.0,
        rician_noise_sigma_range: Tuple[float, float] = (0.01, 0.05),
        inhomogeneity_strength: float = 0.3,
        apply_probability: float = 0.5
    ):
        """
        Initialize physics-informed augmentation.
        
        Args:
            elastic_alpha_range: Range for elastic deformation intensity
            elastic_sigma: Gaussian filter sigma for elastic deformation
            rician_noise_sigma_range: Range for Rician noise sigma
            inhomogeneity_strength: Strength of intensity inhomogeneity
            apply_probability: Probability of applying each augmentation
        """
        self.elastic_alpha_range = elastic_alpha_range
        self.elastic_sigma = elastic_sigma
        self.rician_noise_sigma_range = rician_noise_sigma_range
        self.inhomogeneity_strength = inhomogeneity_strength
        self.apply_probability = apply_probability
        
    def elastic_deformation(self, image: np.ndarray, alpha: float = None) -> np.ndarray:
        """
        Apply elastic deformation to simulate tissue deformation.
        
        Elastic deformation simulates natural tissue movement and deformation
        that occurs in MRI scans due to breathing, cardiac motion, etc.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            alpha: Deformation intensity (random if None)
            
        Returns:
            Deformed image
        """
        if not SCIPY_AVAILABLE:
            return image
            
        if alpha is None:
            alpha = np.random.uniform(*self.elastic_alpha_range)
        
        shape = image.shape[:2]
        
        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * alpha
        
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1),
            self.elastic_sigma
        ) * alpha
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        
        # Apply displacement
        indices = [
            np.clip(y + dy, 0, shape[0] - 1).astype(np.float32),
            np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
        ]
        
        # Apply to each channel if multi-channel
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = map_coordinates(
                    image[:, :, c],
                    indices,
                    order=1,
                    mode='reflect'
                )
            return result
        else:
            return map_coordinates(image, indices, order=1, mode='reflect')
    
    def rician_noise(self, image: np.ndarray, sigma: float = None) -> np.ndarray:
        """
        Add Rician noise to simulate MRI acquisition noise.
        
        Rician noise is the characteristic noise model for magnitude MRI images,
        arising from the quadrature detection in complex k-space.
        
        Args:
            image: Input image (normalized to [0, 1])
            sigma: Noise standard deviation (random if None)
            
        Returns:
            Noisy image
        """
        if sigma is None:
            sigma = np.random.uniform(*self.rician_noise_sigma_range)
        
        # Generate noise for real and imaginary components
        noise_real = np.random.normal(0, sigma, image.shape)
        noise_imag = np.random.normal(0, sigma, image.shape)
        
        # Rician distribution: magnitude of complex signal with Gaussian noise
        # |S + n_r + i*n_i| where S is the signal
        noisy = np.sqrt((image + noise_real)**2 + noise_imag**2)
        
        # Clip to valid range
        return np.clip(noisy, 0, 1)
    
    def intensity_inhomogeneity(self, image: np.ndarray, strength: float = None) -> np.ndarray:
        """
        Apply intensity inhomogeneity to simulate RF coil bias field.
        
        In MRI, the sensitivity of RF coils varies spatially, creating
        a smooth intensity variation across the image (bias field).
        
        Args:
            image: Input image
            strength: Inhomogeneity strength (uses default if None)
            
        Returns:
            Image with intensity inhomogeneity
        """
        if strength is None:
            strength = self.inhomogeneity_strength
        
        shape = image.shape[:2]
        
        # Generate smooth bias field using low-frequency sinusoids
        x = np.linspace(-np.pi, np.pi, shape[1])
        y = np.linspace(-np.pi, np.pi, shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Random phase and frequency
        freq_x = np.random.uniform(0.5, 2.0)
        freq_y = np.random.uniform(0.5, 2.0)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        
        # Create smooth bias field
        bias_field = 1.0 + strength * (
            0.5 * np.sin(freq_x * X + phase_x) +
            0.5 * np.sin(freq_y * Y + phase_y)
        )
        
        # Apply to image
        if len(image.shape) == 3:
            bias_field = bias_field[:, :, np.newaxis]
        
        return np.clip(image * bias_field, 0, 1)
    
    def ghosting_artifact(self, image: np.ndarray, ghost_intensity: float = 0.1) -> np.ndarray:
        """
        Add ghosting artifacts to simulate motion-induced replicas.
        
        Ghosting appears as faint copies of the image shifted in the
        phase-encoding direction, caused by periodic motion during acquisition.
        
        Args:
            image: Input image
            ghost_intensity: Intensity of ghost relative to original
            
        Returns:
            Image with ghosting artifacts
        """
        shape = image.shape[:2]
        
        # Shift amount (typically in phase-encoding direction)
        shift = np.random.randint(shape[0] // 8, shape[0] // 4)
        
        # Create shifted copy (ghost)
        ghost = np.roll(image, shift, axis=0)
        
        # Add ghost to original
        result = image + ghost_intensity * ghost
        
        return np.clip(result, 0, 1)
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random physics-informed augmentations.
        
        Args:
            image: Input image (H, W) or (H, W, C), normalized to [0, 1]
            
        Returns:
            Augmented image
        """
        result = image.copy()
        
        # Apply each augmentation with probability
        if np.random.random() < self.apply_probability:
            result = self.elastic_deformation(result)
        
        if np.random.random() < self.apply_probability:
            result = self.rician_noise(result)
        
        if np.random.random() < self.apply_probability * 0.7:  # Less frequent
            result = self.intensity_inhomogeneity(result)
        
        if np.random.random() < self.apply_probability * 0.3:  # Rare
            result = self.ghosting_artifact(result)
        
        return result.astype(np.float32)


def create_physics_augmentation_layer(
    elastic_alpha_range: Tuple[float, float] = (30, 40),
    rician_noise_sigma_range: Tuple[float, float] = (0.01, 0.05),
    apply_probability: float = 0.5
):
    """
    Factory function to create a physics-informed augmentation layer.
    
    Args:
        elastic_alpha_range: Range for elastic deformation intensity
        rician_noise_sigma_range: Range for Rician noise sigma
        apply_probability: Probability of applying augmentation
        
    Returns:
        PhysicsInformedAugmentation instance
    """
    return PhysicsInformedAugmentation(
        elastic_alpha_range=elastic_alpha_range,
        rician_noise_sigma_range=rician_noise_sigma_range,
        apply_probability=apply_probability
    )


class PhysicsAugmentationGenerator:
    """
    Data generator wrapper that applies physics-informed augmentation.
    """
    
    def __init__(self, base_generator, augmentor: PhysicsInformedAugmentation):
        """
        Initialize generator with physics augmentation.
        
        Args:
            base_generator: Base Keras ImageDataGenerator or tf.data.Dataset
            augmentor: PhysicsInformedAugmentation instance
        """
        self.base_generator = base_generator
        self.augmentor = augmentor
        
    def __iter__(self):
        return self
        
    def __next__(self):
        batch_x, batch_y = next(self.base_generator)
        
        # Apply physics augmentation to each image in batch
        augmented_batch = np.zeros_like(batch_x)
        for i in range(len(batch_x)):
            augmented_batch[i] = self.augmentor.augment(batch_x[i])
        
        return augmented_batch, batch_y
    
    def __len__(self):
        return len(self.base_generator)


if __name__ == "__main__":
    print("Testing Physics-Informed Augmentation...")
    
    # Create test image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    # Create augmentor
    augmentor = PhysicsInformedAugmentation(
        elastic_alpha_range=(30, 40),
        rician_noise_sigma_range=(0.01, 0.05),
        apply_probability=0.8
    )
    
    # Test individual augmentations
    print("\n1. Testing Elastic Deformation...")
    deformed = augmentor.elastic_deformation(test_image)
    print(f"   Input shape: {test_image.shape}, Output shape: {deformed.shape}")
    
    print("\n2. Testing Rician Noise...")
    noisy = augmentor.rician_noise(test_image)
    print(f"   Input shape: {test_image.shape}, Output shape: {noisy.shape}")
    
    print("\n3. Testing Intensity Inhomogeneity...")
    inhomogeneous = augmentor.intensity_inhomogeneity(test_image)
    print(f"   Input shape: {test_image.shape}, Output shape: {inhomogeneous.shape}")
    
    print("\n4. Testing Combined Augmentation...")
    augmented = augmentor.augment(test_image)
    print(f"   Input shape: {test_image.shape}, Output shape: {augmented.shape}")
    
    print("\n✓ Physics-Informed Augmentation test passed!")
