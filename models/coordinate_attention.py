"""
Coordinate Attention Module for Phoenix Protocol.
Position-preserving attention mechanism for medical imaging.

Unlike SEVector which destroys spatial information via global average pooling,
Coordinate Attention preserves position information critical for:
- Tumor location (diagnostic relevance)
- Boundary delineation
- Multi-focal lesion detection

Reference: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CoordinateAttentionBlock(layers.Layer):
    """
    Coordinate Attention Block.
    
    Encodes channel relationships and long-range dependencies with precise
    positional information, unlike SE blocks that lose spatial information.
    
    Key advantages for medical imaging:
    1. Preserves spatial position (tumor location matters)
    2. Captures long-range dependencies (mass effect)
    3. Lightweight (suitable for edge deployment)
    """
    
    def __init__(
        self,
        filters: int,
        reduction_ratio: int = 8,
        **kwargs
    ):
        """
        Initialize Coordinate Attention block.
        
        Args:
            filters: Number of input/output filters
            reduction_ratio: Channel reduction ratio for bottleneck
        """
        super(CoordinateAttentionBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(filters // reduction_ratio, 8)
        
    def build(self, input_shape):
        """Build the layer."""
        # Shared 1x1 convolution for channel reduction
        self.conv_reduce = layers.Conv2D(
            self.reduced_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name='conv_reduce'
        )
        self.bn = layers.BatchNormalization(name='bn')
        
        # Separate convolutions for height and width attention
        self.conv_h = layers.Conv2D(
            self.filters,
            kernel_size=1,
            padding='same',
            name='conv_h'
        )
        self.conv_w = layers.Conv2D(
            self.filters,
            kernel_size=1,
            padding='same',
            name='conv_w'
        )
        
        super(CoordinateAttentionBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor (batch, height, width, channels)
            training: Training mode flag
            
        Returns:
            Attention-weighted output tensor
        """
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Pool along width to get height attention: (B, H, 1, C)
        x_h = tf.reduce_mean(inputs, axis=2, keepdims=True)
        
        # Pool along height to get width attention: (B, 1, W, C)
        x_w = tf.reduce_mean(inputs, axis=1, keepdims=True)
        
        # Transpose x_w for concatenation: (B, W, 1, C)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])
        
        # Concatenate along spatial dimension: (B, H+W, 1, C)
        y = tf.concat([x_h, x_w], axis=1)
        
        # Reduce channels
        y = self.conv_reduce(y)
        y = self.bn(y, training=training)
        y = tf.nn.relu(y)
        
        # Split back into height and width components
        x_h, x_w = tf.split(y, [height, width], axis=1)
        
        # Transpose x_w back: (B, 1, W, reduced_C)
        x_w = tf.transpose(x_w, perm=[0, 2, 1, 3])
        
        # Generate attention weights
        a_h = tf.nn.sigmoid(self.conv_h(x_h))  # (B, H, 1, C)
        a_w = tf.nn.sigmoid(self.conv_w(x_w))  # (B, 1, W, C)
        
        # Apply attention
        output = inputs * a_h * a_w
        
        return output
    
    def get_config(self):
        """Return layer configuration."""
        config = {
            'filters': self.filters,
            'reduction_ratio': self.reduction_ratio
        }
        base_config = super(CoordinateAttentionBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CoordinateAttentionConvBlock(layers.Layer):
    """
    Convolution block with integrated Coordinate Attention.
    Combines convolution, normalization, activation, and attention.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        reduction_ratio: int = 8,
        activation: str = 'relu',
        **kwargs
    ):
        """
        Initialize Coordinate Attention Conv Block.
        
        Args:
            filters: Number of output filters
            kernel_size: Convolution kernel size
            strides: Convolution stride
            reduction_ratio: Attention reduction ratio
            activation: Activation function
        """
        super(CoordinateAttentionConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.reduction_ratio = reduction_ratio
        self.activation_name = activation
        
    def build(self, input_shape):
        """Build the layer."""
        self.conv = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation(self.activation_name)
        self.ca = CoordinateAttentionBlock(
            filters=self.filters,
            reduction_ratio=self.reduction_ratio
        )
        
        super(CoordinateAttentionConvBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.ca(x, training=training)
        return x
    
    def get_config(self):
        """Return configuration."""
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'reduction_ratio': self.reduction_ratio,
            'activation': self.activation_name
        }
        base_config = super(CoordinateAttentionConvBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    print("Testing Coordinate Attention...")
    
    # Create test input
    test_input = tf.random.normal([2, 56, 56, 64])
    
    # Test Coordinate Attention Block
    ca_block = CoordinateAttentionBlock(filters=64, reduction_ratio=8)
    output = ca_block(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {ca_block.count_params():,}")
    
    # Test Coordinate Attention Conv Block
    ca_conv_block = CoordinateAttentionConvBlock(filters=128, kernel_size=3)
    output = ca_conv_block(test_input)
    
    print(f"\nCA Conv Block output shape: {output.shape}")
    print(f"CA Conv Block parameters: {ca_conv_block.count_params():,}")
    
    print("\n✓ Coordinate Attention test passed!")
