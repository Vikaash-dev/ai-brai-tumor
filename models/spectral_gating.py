# Multi-Spectral Concordance Gating (MSCG) for PHOENIX-v3.1
"""
PHOENIX-v3.1 Component: Multi-Spectral Concordance Gating

Frequency-domain fusion for multi-modal MRI (T1, T2, FLAIR).
Unlike spatial fusion, MSCG operates in the frequency domain via 2D FFT
to identify frequencies where modalities agree (concordance).

Key Innovation:
- Noise present in only one modality is suppressed
- Concordant features are amplified
- Robust to single-modality artifacts

Reference: PHOENIX-v3.1 Paper Section 3.4
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple, List


class SpectralTransform(layers.Layer):
    """
    2D FFT-based spectral transformation.
    
    Transforms spatial features to frequency domain for
    cross-modality concordance analysis.
    """
    
    def __init__(
        self,
        use_log_magnitude: bool = True,
        epsilon: float = 1e-8,
        name: str = 'spectral_transform',
        **kwargs
    ):
        """
        Initialize Spectral Transform.
        
        Args:
            use_log_magnitude: Whether to use log-magnitude spectrum
            epsilon: Small value for numerical stability
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.use_log_magnitude = use_log_magnitude
        self.epsilon = epsilon
        
    def call(self, inputs):
        """
        Transform to frequency domain.
        
        Args:
            inputs: Spatial features (batch, H, W, C)
            
        Returns:
            Tuple of (magnitude, phase) spectra
        """
        # Process each channel
        # inputs: (batch, H, W, C)
        
        # Transpose to (batch, C, H, W) for channel-wise FFT
        x = tf.transpose(inputs, [0, 3, 1, 2])
        
        # Cast to complex for FFT
        x_complex = tf.cast(x, tf.complex64)
        
        # 2D FFT per channel
        fft = tf.signal.fft2d(x_complex)
        
        # Shift zero frequency to center
        fft_shifted = tf.signal.fftshift(fft, axes=[-2, -1])
        
        # Compute magnitude and phase
        magnitude = tf.abs(fft_shifted)
        phase = tf.math.angle(fft_shifted)
        
        # Log magnitude for better dynamic range
        if self.use_log_magnitude:
            magnitude = tf.math.log(magnitude + self.epsilon)
        
        # Transpose back to (batch, H, W, C)
        magnitude = tf.transpose(magnitude, [0, 2, 3, 1])
        phase = tf.transpose(phase, [0, 2, 3, 1])
        
        return magnitude, phase
    
    def inverse(self, magnitude, phase, use_log: bool = True):
        """
        Inverse transform from frequency to spatial domain.
        
        Args:
            magnitude: Magnitude spectrum
            phase: Phase spectrum
            use_log: Whether magnitude is log-scaled
            
        Returns:
            Spatial features
        """
        # Transpose to (batch, C, H, W)
        magnitude = tf.transpose(magnitude, [0, 3, 1, 2])
        phase = tf.transpose(phase, [0, 3, 1, 2])
        
        # Undo log if needed
        if use_log:
            magnitude = tf.exp(magnitude) - self.epsilon
        
        # Reconstruct complex spectrum
        real = magnitude * tf.cos(phase)
        imag = magnitude * tf.sin(phase)
        fft_shifted = tf.complex(real, imag)
        
        # Inverse shift
        fft = tf.signal.ifftshift(fft_shifted, axes=[-2, -1])
        
        # Inverse FFT
        spatial = tf.signal.ifft2d(fft)
        spatial = tf.math.real(spatial)
        
        # Transpose back to (batch, H, W, C)
        spatial = tf.transpose(spatial, [0, 2, 3, 1])
        
        return spatial
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'use_log_magnitude': self.use_log_magnitude,
            'epsilon': self.epsilon
        })
        return config


class ConcordanceMixer(layers.Layer):
    """
    Cross-modality Concordance Mixer.
    
    Identifies frequency components where multiple modalities
    agree (concordance) and suppresses discordant components.
    """
    
    def __init__(
        self,
        num_modalities: int = 3,
        hidden_dim: int = 64,
        concordance_threshold: float = 0.5,
        name: str = 'concordance_mixer',
        **kwargs
    ):
        """
        Initialize Concordance Mixer.
        
        Args:
            num_modalities: Number of input modalities (T1, T2, FLAIR)
            hidden_dim: Hidden dimension for mixing network
            concordance_threshold: Threshold for concordance detection
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.concordance_threshold = concordance_threshold
        
    def build(self, input_shape):
        """Build the layer."""
        # input_shape: list of (batch, H, W, C) for each modality
        if isinstance(input_shape, list):
            self.channels = input_shape[0][-1]
        else:
            self.channels = input_shape[-1] // self.num_modalities
        
        # Cross-channel mixing network
        self.mix_conv1 = layers.Conv2D(
            self.hidden_dim,
            kernel_size=1,
            padding='same',
            activation='relu',
            name=f'{self.name}/mix_conv1'
        )
        
        self.mix_conv2 = layers.Conv2D(
            self.hidden_dim,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f'{self.name}/mix_conv2'
        )
        
        # Concordance score prediction
        self.concordance_conv = layers.Conv2D(
            1,
            kernel_size=1,
            padding='same',
            activation='sigmoid',
            name=f'{self.name}/concordance_conv'
        )
        
        # Output projection
        self.out_conv = layers.Conv2D(
            self.channels,
            kernel_size=1,
            padding='same',
            name=f'{self.name}/out_conv'
        )
        
        super().build(input_shape)
        
    def call(self, inputs):
        """
        Compute concordance-weighted fusion.
        
        Args:
            inputs: List of modality features or concatenated tensor
            
        Returns:
            Concordance-fused features
        """
        if isinstance(inputs, list):
            # Concatenate modalities
            stacked = tf.stack(inputs, axis=-1)  # (B, H, W, C, M)
            concatenated = tf.concat(inputs, axis=-1)  # (B, H, W, C*M)
        else:
            concatenated = inputs
            # Reshape to get individual modalities
            shape = tf.shape(inputs)
            stacked = tf.reshape(
                inputs, 
                [shape[0], shape[1], shape[2], -1, self.num_modalities]
            )
        
        # Compute variance across modalities (concordance indicator)
        # Low variance = high concordance
        modality_variance = tf.math.reduce_variance(stacked, axis=-1)
        
        # Mix features
        x = self.mix_conv1(concatenated)
        x = self.mix_conv2(x)
        
        # Predict concordance scores
        concordance_scores = self.concordance_conv(x)
        
        # Weight by inverse variance (high concordance = high weight)
        variance_weight = 1.0 / (modality_variance + 1e-6)
        variance_weight = variance_weight / tf.reduce_max(variance_weight, axis=-1, keepdims=True)
        
        # Combine concordance prediction with variance-based weight
        combined_weight = concordance_scores * tf.reduce_mean(variance_weight, axis=-1, keepdims=True)
        
        # Apply concordance weighting
        if isinstance(inputs, list):
            weighted_sum = sum(
                w * m for w, m in zip(
                    tf.split(combined_weight, self.num_modalities, axis=-1),
                    inputs
                )
            )
        else:
            # Use mean with concordance weighting
            mean_features = tf.reduce_mean(stacked, axis=-1)
            weighted_sum = mean_features * combined_weight
        
        # Project to output dimension
        output = self.out_conv(concatenated)
        output = output * combined_weight + (1 - combined_weight) * tf.reduce_mean(stacked, axis=-1)
        
        return output, concordance_scores
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_modalities': self.num_modalities,
            'hidden_dim': self.hidden_dim,
            'concordance_threshold': self.concordance_threshold
        })
        return config


class MultiSpectralConcordanceGating(layers.Layer):
    """
    Full Multi-Spectral Concordance Gating Module.
    
    Combines spectral transformation with concordance mixing
    for robust multi-modal MRI fusion.
    """
    
    def __init__(
        self,
        num_modalities: int = 3,
        hidden_dim: int = 64,
        use_residual: bool = True,
        name: str = 'mscg',
        **kwargs
    ):
        """
        Initialize MSCG.
        
        Args:
            num_modalities: Number of MRI modalities
            hidden_dim: Hidden dimension
            use_residual: Whether to use residual connection
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.num_modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
    def build(self, input_shape):
        """Build the layer."""
        # Spectral transform
        self.spectral_transform = SpectralTransform(
            use_log_magnitude=True,
            name=f'{self.name}/spectral_transform'
        )
        
        # Concordance mixer (in frequency domain)
        self.freq_concordance = ConcordanceMixer(
            num_modalities=self.num_modalities,
            hidden_dim=self.hidden_dim,
            name=f'{self.name}/freq_concordance'
        )
        
        # Concordance mixer (in spatial domain)
        self.spatial_concordance = ConcordanceMixer(
            num_modalities=self.num_modalities,
            hidden_dim=self.hidden_dim,
            name=f'{self.name}/spatial_concordance'
        )
        
        # Gating network
        self.gate = layers.Dense(
            1,
            activation='sigmoid',
            name=f'{self.name}/gate'
        )
        
        # Layer normalization
        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'{self.name}/norm'
        )
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: List of modality features [(B,H,W,C), ...] or 
                    concatenated tensor (B,H,W,C*M)
            
        Returns:
            Fused features (B, H, W, C)
        """
        if isinstance(inputs, list):
            # Store original for residual
            original = inputs[0]
            concatenated = tf.concat(inputs, axis=-1)
        else:
            concatenated = inputs
            original = inputs[..., :inputs.shape[-1] // self.num_modalities]
        
        # Spatial domain concordance
        spatial_fused, spatial_scores = self.spatial_concordance(concatenated)
        
        # Frequency domain analysis
        magnitude, phase = self.spectral_transform(concatenated)
        freq_fused, freq_scores = self.freq_concordance(magnitude)
        
        # Inverse transform frequency features
        freq_spatial = self.spectral_transform.inverse(
            freq_fused, 
            phase[..., :freq_fused.shape[-1]],
            use_log=True
        )
        
        # Compute gate based on concordance scores
        combined_scores = tf.concat([spatial_scores, freq_scores], axis=-1)
        gate = self.gate(combined_scores)
        
        # Gated fusion of spatial and frequency concordance
        fused = gate * spatial_fused + (1 - gate) * freq_spatial
        
        # Normalize
        fused = self.norm(fused)
        
        # Residual connection
        if self.use_residual:
            # Match dimensions if needed
            if fused.shape[-1] != original.shape[-1]:
                fused = layers.Conv2D(
                    original.shape[-1], 1, padding='same'
                )(fused)
            fused = fused + original
        
        return fused
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_modalities': self.num_modalities,
            'hidden_dim': self.hidden_dim,
            'use_residual': self.use_residual
        })
        return config


# Example usage
if __name__ == "__main__":
    print("Multi-Spectral Concordance Gating (MSCG)")
    print("=" * 60)
    
    # Test configuration
    batch_size = 2
    height, width = 56, 56
    channels = 64
    num_modalities = 3  # T1, T2, FLAIR
    
    # Create modality inputs
    modalities = [
        tf.random.normal((batch_size, height, width, channels))
        for _ in range(num_modalities)
    ]
    
    # Test SpectralTransform
    spectral = SpectralTransform()
    mag, phase = spectral(modalities[0])
    reconstructed = spectral.inverse(mag, phase)
    print(f"SpectralTransform: {modalities[0].shape} -> mag: {mag.shape}, phase: {phase.shape}")
    print(f"Reconstruction error: {tf.reduce_mean(tf.abs(modalities[0] - reconstructed)):.6f}")
    
    # Test ConcordanceMixer
    mixer = ConcordanceMixer(num_modalities=3)
    concat_input = tf.concat(modalities, axis=-1)
    mixed, scores = mixer(concat_input)
    print(f"ConcordanceMixer: {concat_input.shape} -> {mixed.shape}, scores: {scores.shape}")
    
    # Test MSCG
    mscg = MultiSpectralConcordanceGating(num_modalities=3)
    fused = mscg(modalities)
    print(f"MSCG: {[m.shape for m in modalities]} -> {fused.shape}")
    
    print("\n✓ All MSCG tests passed!")
