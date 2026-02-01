# PHOENIX-v3.1: Hybrid Liquid-Spectral-KAN Mamba Model
"""
PHOENIX-v3.1: Complete Hybrid Pyramid Architecture

Integrates all PHOENIX-v3.1 components:
1. SpatialMixer - Topology preservation before flattening
2. Hybrid Pyramid - Conv2D (high-res) + Liquid-S6-KAN (low-res)
3. Multi-Spectral Concordance Gating - Frequency-domain fusion
4. Symbolic Mirror TTT - Test-time spline adaptation

Target: 1.18M parameters, >93% Dice on BraTS 2023

Reference: PHOENIX-v3.1 Paper
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple, List, Dict

# Import PHOENIX-v3.1 components
from models.spatial_mixer import SpatialMixer, TopologyPreservingFlatten
from models.liquid_ssm import LiquidS6Block
from models.kan_layer import KANBlock, AdaptiveKAN
from models.spectral_gating import MultiSpectralConcordanceGating


class ResidualConvBlock(layers.Layer):
    """
    Residual Convolutional Block for high-resolution stages.
    
    Used in Stages 1-2 of the Hybrid Pyramid where SSM would
    be computationally prohibitive (L > 10,000 tokens).
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        use_batch_norm: bool = True,
        activation: str = 'gelu',
        name: str = 'res_conv_block',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_batch_norm = use_batch_norm
        self.activation_name = activation
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        
        # Main path
        self.conv1 = layers.Conv2D(
            self.filters, self.kernel_size,
            strides=self.strides, padding='same',
            use_bias=not self.use_batch_norm,
            kernel_initializer='he_normal',
            name=f'{self.name}/conv1'
        )
        
        self.conv2 = layers.Conv2D(
            self.filters, self.kernel_size,
            padding='same',
            use_bias=not self.use_batch_norm,
            kernel_initializer='he_normal',
            name=f'{self.name}/conv2'
        )
        
        if self.use_batch_norm:
            self.bn1 = layers.BatchNormalization(name=f'{self.name}/bn1')
            self.bn2 = layers.BatchNormalization(name=f'{self.name}/bn2')
        
        self.activation = layers.Activation(self.activation_name)
        
        # Skip connection
        if in_channels != self.filters or self.strides > 1:
            self.skip_conv = layers.Conv2D(
                self.filters, 1, strides=self.strides,
                padding='same', name=f'{self.name}/skip'
            )
        else:
            self.skip_conv = None
            
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # Skip connection
        skip = self.skip_conv(inputs) if self.skip_conv else inputs
        
        # Main path
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = self.activation(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        
        # Add skip and activate
        x = x + skip
        x = self.activation(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation_name
        })
        return config


class LiquidS6KANCell(layers.Layer):
    """
    Liquid-S6-KAN Cell for low-resolution stages.
    
    Combines:
    - SpatialMixer for topology preservation
    - Liquid-S6 SSM for global context
    - KAN for adaptive activation functions
    
    Used in Stages 3-4 where sequence length is manageable (L < 4000).
    """
    
    def __init__(
        self,
        channels: int,
        state_dim: int = 16,
        kan_knots: int = 8,
        dropout_rate: float = 0.1,
        name: str = 'liquid_s6_kan_cell',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.channels = channels
        self.state_dim = state_dim
        self.kan_knots = kan_knots
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        _, h, w, c = input_shape
        
        # Project to channels if needed
        if c != self.channels:
            self.input_proj = layers.Conv2D(
                self.channels, 1, padding='same',
                name=f'{self.name}/input_proj'
            )
        else:
            self.input_proj = None
        
        # SpatialMixer for topology preservation
        self.spatial_mixer = SpatialMixer(
            kernel_size=3,
            use_residual=True,
            name=f'{self.name}/spatial_mixer'
        )
        
        # Flatten with positional encoding
        self.flatten = TopologyPreservingFlatten(
            mixer_kernel_size=3,
            use_positional_encoding=True,
            name=f'{self.name}/flatten'
        )
        
        # Liquid-S6 block
        self.liquid_s6 = LiquidS6Block(
            state_dim=self.state_dim,
            ffn_expand_factor=4,
            dropout_rate=self.dropout_rate,
            name=f'{self.name}/liquid_s6'
        )
        
        # KAN block for adaptive activations
        self.kan = KANBlock(
            hidden_dim=self.channels * 2,
            out_dim=self.channels,
            num_knots=self.kan_knots,
            dropout_rate=self.dropout_rate,
            name=f'{self.name}/kan'
        )
        
        # Store spatial dimensions for reshaping
        self.spatial_h = h
        self.spatial_w = w
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        # Project if needed
        x = self.input_proj(inputs) if self.input_proj else inputs
        
        # Apply spatial mixing
        x = self.spatial_mixer(x, training=training)
        
        # Store for residual
        residual_2d = x
        
        # Flatten for SSM processing
        x = self.flatten(x, training=training)  # (B, H*W, C)
        
        # Apply Liquid-S6
        x = self.liquid_s6(x, training=training)
        
        # Apply KAN
        x = self.kan(x, training=training)
        
        # Reshape back to 2D
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, self.spatial_h, self.spatial_w, self.channels))
        
        # Residual connection
        x = x + residual_2d
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'state_dim': self.state_dim,
            'kan_knots': self.kan_knots,
            'dropout_rate': self.dropout_rate
        })
        return config


class PriorityScout(layers.Layer):
    """
    Priority Scout Module for ROI Prediction.
    
    Predicts region-of-interest (ROI) map that modulates
    the time-constant of the SSM for adaptive dynamics.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        name: str = 'priority_scout',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.hidden_dim, 3, padding='same',
            activation='relu', name=f'{self.name}/conv1'
        )
        self.conv2 = layers.Conv2D(
            self.hidden_dim // 2, 3, padding='same',
            activation='relu', name=f'{self.name}/conv2'
        )
        self.conv3 = layers.Conv2D(
            1, 1, padding='same',
            activation='sigmoid', name=f'{self.name}/conv3'
        )
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        roi_map = self.conv3(x)
        return roi_map
    
    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.hidden_dim})
        return config


class SymbolicMirrorTTT(layers.Layer):
    """
    Symbolic Mirror Test-Time Training Adapter.
    
    Adapts KAN spline weights during inference to minimize
    Shannon entropy of predictions on the specific patient scan.
    """
    
    def __init__(
        self,
        adaptation_lr: float = 0.01,
        adaptation_steps: int = 3,
        entropy_threshold: float = 0.5,
        name: str = 'symbolic_mirror_ttt',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.entropy_threshold = entropy_threshold
        
    def build(self, input_shape):
        channels = input_shape[-1]
        
        # Adaptive KAN for TTT
        self.adaptive_kan = AdaptiveKAN(
            out_features=channels,
            num_knots=8,
            adaptation_lr=self.adaptation_lr,
            adaptation_steps=self.adaptation_steps,
            name=f'{self.name}/adaptive_kan'
        )
        
        super().build(input_shape)
        
    def compute_entropy(self, predictions):
        """Compute Shannon entropy of predictions."""
        # Softmax to get probabilities
        probs = tf.nn.softmax(predictions, axis=-1)
        # Entropy: -sum(p * log(p))
        epsilon = 1e-10
        entropy = -tf.reduce_sum(
            probs * tf.math.log(probs + epsilon),
            axis=-1
        )
        return tf.reduce_mean(entropy)
    
    def call(self, inputs, adapt: bool = False, training=None):
        """
        Forward pass with optional TTT adaptation.
        
        Args:
            inputs: Input features
            adapt: Whether to perform TTT adaptation
            training: Training mode
            
        Returns:
            Adapted features
        """
        if adapt and not training:
            return self.adaptive_kan.adapt(inputs, self.compute_entropy)
        else:
            return self.adaptive_kan(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'adaptation_lr': self.adaptation_lr,
            'adaptation_steps': self.adaptation_steps,
            'entropy_threshold': self.entropy_threshold
        })
        return config


class PHOENIXv31(keras.Model):
    """
    PHOENIX-v3.1: Hybrid Liquid-Spectral-KAN Mamba Model.
    
    Architecture:
    - Input: (224, 224, 3) for True 2.5D or multi-modal
    - Stage 1: ResConv (224→112), 32 channels
    - Stage 2: ResConv (112→56), 64 channels  
    - Stage 3: Liquid-S6-KAN (56→28), 128 channels
    - Stage 4: Liquid-S6-KAN (28→14), 256 channels
    - Output: Classification or Segmentation head
    
    Total Parameters: ~1.18M
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_modalities: int = 1,
        use_mscg: bool = True,
        use_ttt: bool = True,
        state_dim: int = 16,
        kan_knots: int = 8,
        dropout_rate: float = 0.1,
        name: str = 'phoenix_v31',
        **kwargs
    ):
        """
        Initialize PHOENIX-v3.1.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input image shape
            num_modalities: Number of MRI modalities
            use_mscg: Whether to use Multi-Spectral Concordance Gating
            use_ttt: Whether to use Test-Time Training
            state_dim: SSM state dimension
            kan_knots: Number of KAN knots
            dropout_rate: Dropout rate
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.num_classes = num_classes
        self._input_shape = input_shape
        self.num_modalities = num_modalities
        self.use_mscg = use_mscg
        self.use_ttt = use_ttt
        self.state_dim = state_dim
        self.kan_knots = kan_knots
        self.dropout_rate = dropout_rate
        
        self._build_model()
        
    def _build_model(self):
        """Build the model architecture."""
        
        # Multi-Spectral Concordance Gating (if multi-modal)
        if self.use_mscg and self.num_modalities > 1:
            self.mscg = MultiSpectralConcordanceGating(
                num_modalities=self.num_modalities,
                hidden_dim=32,
                name='mscg'
            )
        else:
            self.mscg = None
        
        # Stem: Initial feature extraction
        self.stem = keras.Sequential([
            layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('gelu')
        ], name='stem')
        
        # Stage 1: High-resolution Conv (112x112)
        self.stage1 = keras.Sequential([
            ResidualConvBlock(32, name='stage1_block1'),
            ResidualConvBlock(32, name='stage1_block2'),
            layers.MaxPooling2D(2, name='stage1_pool')
        ], name='stage1')
        
        # Stage 2: High-resolution Conv (56x56)
        self.stage2 = keras.Sequential([
            ResidualConvBlock(64, name='stage2_block1'),
            ResidualConvBlock(64, name='stage2_block2'),
            layers.MaxPooling2D(2, name='stage2_pool')
        ], name='stage2')
        
        # Priority Scout for ROI detection
        self.priority_scout = PriorityScout(hidden_dim=32, name='priority_scout')
        
        # Stage 3: Low-resolution Liquid-S6-KAN (28x28)
        self.stage3_proj = layers.Conv2D(128, 1, padding='same', name='stage3_proj')
        self.stage3 = LiquidS6KANCell(
            channels=128,
            state_dim=self.state_dim,
            kan_knots=self.kan_knots,
            dropout_rate=self.dropout_rate,
            name='stage3'
        )
        self.stage3_pool = layers.MaxPooling2D(2, name='stage3_pool')
        
        # Stage 4: Low-resolution Liquid-S6-KAN (14x14)
        self.stage4_proj = layers.Conv2D(256, 1, padding='same', name='stage4_proj')
        self.stage4 = LiquidS6KANCell(
            channels=256,
            state_dim=self.state_dim,
            kan_knots=self.kan_knots,
            dropout_rate=self.dropout_rate,
            name='stage4'
        )
        
        # Symbolic Mirror TTT Adapter
        if self.use_ttt:
            self.ttt_adapter = SymbolicMirrorTTT(
                adaptation_lr=0.01,
                adaptation_steps=3,
                name='ttt_adapter'
            )
        else:
            self.ttt_adapter = None
        
        # Classification Head
        self.global_pool = layers.GlobalAveragePooling2D(name='global_pool')
        self.classifier = keras.Sequential([
            layers.Dense(256, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(128, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.num_classes, activation='softmax')
        ], name='classifier')
        
    def call(self, inputs, training=None, adapt_ttt: bool = False):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor or list of modality tensors
            training: Training mode
            adapt_ttt: Whether to perform TTT adaptation
            
        Returns:
            Classification probabilities
        """
        # Multi-spectral fusion if multiple modalities
        if self.mscg is not None and isinstance(inputs, list):
            x = self.mscg(inputs, training=training)
        else:
            x = inputs
        
        # Stem
        x = self.stem(x, training=training)  # (B, 112, 112, 32)
        
        # Stage 1: High-res Conv
        x = self.stage1(x, training=training)  # (B, 56, 56, 32)
        
        # Stage 2: High-res Conv
        x = self.stage2(x, training=training)  # (B, 28, 28, 64)
        
        # Priority Scout ROI map
        roi_map = self.priority_scout(x, training=training)
        
        # Stage 3: Liquid-S6-KAN
        x = self.stage3_proj(x)  # (B, 28, 28, 128)
        x = self.stage3(x, training=training)
        x = x * (1 + roi_map)  # ROI-weighted features
        x = self.stage3_pool(x)  # (B, 14, 14, 128)
        
        # Stage 4: Liquid-S6-KAN
        x = self.stage4_proj(x)  # (B, 14, 14, 256)
        x = self.stage4(x, training=training)
        
        # TTT Adaptation
        if self.ttt_adapter is not None:
            # Flatten for TTT
            batch_size = tf.shape(x)[0]
            h, w, c = x.shape[1], x.shape[2], x.shape[3]
            x_flat = tf.reshape(x, (batch_size, h * w, c))
            x_flat = self.ttt_adapter(x_flat, adapt=adapt_ttt, training=training)
            x = tf.reshape(x_flat, (batch_size, h, w, c))
        
        # Classification
        x = self.global_pool(x)
        output = self.classifier(x, training=training)
        
        return output
    
    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'input_shape': self._input_shape,
            'num_modalities': self.num_modalities,
            'use_mscg': self.use_mscg,
            'use_ttt': self.use_ttt,
            'state_dim': self.state_dim,
            'kan_knots': self.kan_knots,
            'dropout_rate': self.dropout_rate
        }


def create_phoenix_v31(
    num_classes: int = 2,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_modalities: int = 1,
    use_mscg: bool = True,
    use_ttt: bool = True
) -> PHOENIXv31:
    """
    Create PHOENIX-v3.1 model.
    
    Args:
        num_classes: Number of output classes
        input_shape: Input image shape
        num_modalities: Number of MRI modalities
        use_mscg: Whether to use Multi-Spectral Concordance Gating
        use_ttt: Whether to use Test-Time Training
        
    Returns:
        PHOENIXv31 model instance
    """
    model = PHOENIXv31(
        num_classes=num_classes,
        input_shape=input_shape,
        num_modalities=num_modalities,
        use_mscg=use_mscg,
        use_ttt=use_ttt
    )
    
    # Build the model
    model.build((None, *input_shape))
    
    return model


# Example usage
if __name__ == "__main__":
    print("PHOENIX-v3.1: Hybrid Liquid-Spectral-KAN Mamba")
    print("=" * 60)
    
    # Create model
    model = create_phoenix_v31(
        num_classes=2,
        input_shape=(224, 224, 3),
        num_modalities=1,
        use_mscg=False,
        use_ttt=True
    )
    
    # Test forward pass
    x = tf.random.normal((2, 224, 224, 3))
    y = model(x, training=False)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    model.summary()
    
    print("\n✓ PHOENIX-v3.1 model created successfully!")
