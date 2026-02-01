# SpatialMixer: Topology-Aware Spatial Mixing Layer
"""
PHOENIX-v3.1 Component: SpatialMixer

Addresses the "Zombie Topology" problem in State-Space Models.
Standard SSMs require flattening 2D feature maps into 1D sequences,
which destroys vertical adjacency (pixel (i,j) connects to (i,j+1) 
but is distant from (i+1,j)).

The SpatialMixer layer is inserted *before* every flattening operation
to ensure every token encodes information from its 8 spatial neighbors.

Reference: PHOENIX-v3.1 Paper Section 3.2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple


class SpatialMixer(layers.Layer):
    """
    Topology-Aware Spatial Mixing Layer.
    
    Preserves 2D adjacency information before sequence flattening by
    applying depthwise convolution to mix spatial neighbors.
    
    Formula: x_mixed = DepthwiseConv2D_3x3(x) + x (residual)
    
    This ensures that every token in the flattened sequence encodes
    information from its 8 spatial neighbors, effectively recovering
    2D topology without the complexity of 4-directional scanning.
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        activation: Optional[str] = 'gelu',
        name: str = 'spatial_mixer',
        **kwargs
    ):
        """
        Initialize SpatialMixer.
        
        Args:
            kernel_size: Size of depthwise convolution kernel (default: 3)
            use_residual: Whether to add residual connection
            use_layer_norm: Whether to apply layer normalization
            activation: Activation function (default: 'gelu')
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.activation_name = activation
        
    def build(self, input_shape):
        """Build the layer."""
        channels = input_shape[-1]
        
        # Depthwise separable convolution for efficient spatial mixing
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            padding='same',
            depth_multiplier=1,
            use_bias=False,
            kernel_initializer='he_normal',
            name=f'{self.name}/depthwise_conv'
        )
        
        # Pointwise convolution to mix channels
        self.pointwise_conv = layers.Conv2D(
            filters=channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name=f'{self.name}/pointwise_conv'
        )
        
        # Layer normalization
        if self.use_layer_norm:
            self.layer_norm = layers.LayerNormalization(
                epsilon=1e-6,
                name=f'{self.name}/layer_norm'
            )
        
        # Activation
        if self.activation_name:
            self.activation = layers.Activation(
                self.activation_name,
                name=f'{self.name}/activation'
            )
        else:
            self.activation = None
            
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor of shape (batch, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Spatially mixed tensor of same shape
        """
        # Store residual
        residual = inputs
        
        # Apply depthwise convolution to mix spatial neighbors
        x = self.depthwise_conv(inputs)
        
        # Apply pointwise convolution to mix channels
        x = self.pointwise_conv(x)
        
        # Apply layer normalization
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        # Apply activation
        if self.activation:
            x = self.activation(x)
        
        # Add residual connection
        if self.use_residual:
            x = x + residual
            
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'use_residual': self.use_residual,
            'use_layer_norm': self.use_layer_norm,
            'activation': self.activation_name
        })
        return config


class MultiScaleSpatialMixer(layers.Layer):
    """
    Multi-Scale Spatial Mixer for capturing different neighborhood sizes.
    
    Uses parallel depthwise convolutions with different kernel sizes
    to capture both fine and coarse spatial relationships.
    """
    
    def __init__(
        self,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        use_residual: bool = True,
        name: str = 'multi_scale_spatial_mixer',
        **kwargs
    ):
        """
        Initialize MultiScaleSpatialMixer.
        
        Args:
            kernel_sizes: Tuple of kernel sizes for multi-scale mixing
            use_residual: Whether to add residual connection
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.kernel_sizes = kernel_sizes
        self.use_residual = use_residual
        
    def build(self, input_shape):
        """Build the layer."""
        channels = input_shape[-1]
        
        # Create depthwise convolutions for each scale
        self.depthwise_convs = []
        for ks in self.kernel_sizes:
            conv = layers.DepthwiseConv2D(
                kernel_size=ks,
                padding='same',
                depth_multiplier=1,
                use_bias=False,
                kernel_initializer='he_normal',
                name=f'{self.name}/depthwise_conv_{ks}x{ks}'
            )
            self.depthwise_convs.append(conv)
        
        # Fusion convolution
        self.fusion_conv = layers.Conv2D(
            filters=channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name=f'{self.name}/fusion_conv'
        )
        
        self.layer_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'{self.name}/layer_norm'
        )
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass."""
        residual = inputs
        
        # Apply multi-scale depthwise convolutions
        multi_scale_features = []
        for conv in self.depthwise_convs:
            multi_scale_features.append(conv(inputs))
        
        # Concatenate and fuse
        x = tf.concat(multi_scale_features, axis=-1)
        x = self.fusion_conv(x)
        x = self.layer_norm(x)
        x = tf.nn.gelu(x)
        
        if self.use_residual:
            x = x + residual
            
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'kernel_sizes': self.kernel_sizes,
            'use_residual': self.use_residual
        })
        return config


class TopologyPreservingFlatten(layers.Layer):
    """
    Topology-Preserving Flatten Layer.
    
    Combines SpatialMixer with flattening to create spatially-aware
    tokens for State-Space Model processing.
    """
    
    def __init__(
        self,
        mixer_kernel_size: int = 3,
        use_positional_encoding: bool = True,
        name: str = 'topology_preserving_flatten',
        **kwargs
    ):
        """
        Initialize TopologyPreservingFlatten.
        
        Args:
            mixer_kernel_size: Kernel size for spatial mixer
            use_positional_encoding: Whether to add 2D positional encoding
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.mixer_kernel_size = mixer_kernel_size
        self.use_positional_encoding = use_positional_encoding
        
    def build(self, input_shape):
        """Build the layer."""
        _, h, w, c = input_shape
        
        # Spatial mixer
        self.spatial_mixer = SpatialMixer(
            kernel_size=self.mixer_kernel_size,
            use_residual=True,
            name=f'{self.name}/spatial_mixer'
        )
        
        # Learnable 2D positional encoding
        if self.use_positional_encoding:
            self.pos_encoding = self.add_weight(
                name='positional_encoding',
                shape=(1, h, w, c),
                initializer='zeros',
                trainable=True
            )
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor of shape (batch, height, width, channels)
            
        Returns:
            Flattened tensor of shape (batch, height*width, channels)
        """
        # Apply spatial mixing to preserve topology
        x = self.spatial_mixer(inputs, training=training)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = x + self.pos_encoding
        
        # Flatten spatial dimensions
        batch_size = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        x = tf.reshape(x, (batch_size, h * w, c))
        
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'mixer_kernel_size': self.mixer_kernel_size,
            'use_positional_encoding': self.use_positional_encoding
        })
        return config


# Example usage and testing
if __name__ == "__main__":
    print("SpatialMixer: Topology-Aware Spatial Mixing")
    print("=" * 60)
    
    # Test SpatialMixer
    batch_size = 2
    height, width, channels = 56, 56, 128
    
    x = tf.random.normal((batch_size, height, width, channels))
    
    # Test basic SpatialMixer
    mixer = SpatialMixer(kernel_size=3, use_residual=True)
    y = mixer(x)
    print(f"SpatialMixer: {x.shape} -> {y.shape}")
    
    # Test MultiScaleSpatialMixer
    multi_mixer = MultiScaleSpatialMixer(kernel_sizes=(3, 5, 7))
    y_multi = multi_mixer(x)
    print(f"MultiScaleSpatialMixer: {x.shape} -> {y_multi.shape}")
    
    # Test TopologyPreservingFlatten
    flatten = TopologyPreservingFlatten(mixer_kernel_size=3)
    y_flat = flatten(x)
    print(f"TopologyPreservingFlatten: {x.shape} -> {y_flat.shape}")
    
    print("\n✓ All SpatialMixer tests passed!")
