"""
Dynamic Snake Convolutions (DSC) for Phoenix Protocol.
Implements deformable convolutions that can "snake" along curvilinear features.
Critical for tracing irregular, finger-like infiltrations of Glioblastomas.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class DynamicSnakeConv2D(layers.Layer):
    """
    Dynamic Snake Convolution layer.
    
    Unlike standard square kernels, DSC kernels can dynamically deform their shape
    during inference to "snake" along curvilinear features (e.g., tumor boundaries).
    
    This is implemented as a deformable convolution where offset fields are learned
    to adaptively sample from irregular grid positions.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = 'same',
        dilation_rate: int = 1,
        use_bias: bool = True,
        activation: str = None,
        kernel_initializer: str = 'glorot_uniform',
        bias_initializer: str = 'zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        deformable_groups: int = 1,
        **kwargs
    ):
        """
        Initialize Dynamic Snake Convolution layer.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of the convolution kernel
            strides: Stride of the convolution
            padding: Padding mode ('same' or 'valid')
            dilation_rate: Dilation rate for dilated convolution
            use_bias: Whether to use bias
            activation: Activation function
            kernel_initializer: Initializer for kernel weights
            bias_initializer: Initializer for bias
            kernel_regularizer: Regularizer for kernel weights
            bias_regularizer: Regularizer for bias
            deformable_groups: Number of deformable groups for grouped deformation
        """
        super(DynamicSnakeConv2D, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides
        self.padding = padding.upper()
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.deformable_groups = deformable_groups
        
    def build(self, input_shape):
        """Build the layer."""
        input_channels = input_shape[-1]
        
        # Offset prediction network
        # Predicts 2D offsets (dx, dy) for each kernel position
        offset_channels = 2 * self.kernel_size[0] * self.kernel_size[1]
        
        self.offset_conv = layers.Conv2D(
            filters=offset_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding.lower(),
            dilation_rate=self.dilation_rate,
            kernel_initializer='zeros',  # Initialize to zero for identity deformation
            name='offset_conv'
        )
        
        # Modulation (attention) weights for adaptive feature weighting
        self.modulation_conv = layers.Conv2D(
            filters=self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding.lower(),
            dilation_rate=self.dilation_rate,
            activation='sigmoid',
            kernel_initializer='ones',
            name='modulation_conv'
        )
        
        # Main convolution kernel
        self.kernel = self.add_weight(
            name='kernel',
            shape=(*self.kernel_size, input_channels, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True
            )
        
        super(DynamicSnakeConv2D, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass with deformable convolution.
        
        Args:
            inputs: Input tensor (batch, height, width, channels)
            training: Training mode flag
            
        Returns:
            Output tensor after deformable convolution
        """
        # Predict offsets for kernel sampling positions
        offsets = self.offset_conv(inputs)
        
        # Predict modulation weights (adaptive feature importance)
        modulation = self.modulation_conv(inputs)
        
        # Apply deformable convolution
        output = self._deformable_conv2d(inputs, offsets, modulation)
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output
    
    def _deformable_conv2d(self, inputs, offsets, modulation):
        """
        Core deformable convolution operation.
        
        Args:
            inputs: Input feature map
            offsets: Predicted offset fields
            modulation: Modulation weights
            
        Returns:
            Output feature map
        """
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = inputs.shape[3]
        
        # Reshape offsets: (batch, h, w, 2*k*k) -> (batch, h, w, k*k, 2)
        kernel_h, kernel_w = self.kernel_size
        num_points = kernel_h * kernel_w
        
        offsets = tf.reshape(offsets, [batch_size, height, width, num_points, 2])
        modulation = tf.reshape(modulation, [batch_size, height, width, num_points, 1])
        
        # Generate base sampling grid
        base_grid = self._get_base_grid(height, width, kernel_h, kernel_w)
        base_grid = tf.cast(base_grid, tf.float32)
        
        # Add offsets to base grid
        sampling_grid = base_grid + offsets
        
        # Sample features at deformed positions using bilinear interpolation
        sampled_features = self._bilinear_sample(inputs, sampling_grid)
        
        # Apply modulation weights
        sampled_features = sampled_features * modulation
        
        # Reshape for convolution: (batch, h, w, k*k*c)
        sampled_features = tf.reshape(
            sampled_features,
            [batch_size, height, width, num_points * channels]
        )
        
        # Reshape kernel for grouped convolution
        kernel_reshaped = tf.reshape(
            self.kernel,
            [1, 1, num_points * channels, self.filters]
        )
        
        # Apply 1x1 convolution with reshaped kernel
        output = tf.nn.conv2d(
            sampled_features,
            kernel_reshaped,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        
        return output
    
    def _get_base_grid(self, height, width, kernel_h, kernel_w):
        """
        Generate base sampling grid for kernel positions.
        
        Args:
            height: Output height
            width: Output width
            kernel_h: Kernel height
            kernel_w: Kernel width
            
        Returns:
            Base grid tensor
        """
        # Create kernel offset coordinates
        kernel_y = tf.range(-(kernel_h // 2), kernel_h // 2 + 1, dtype=tf.float32)
        kernel_x = tf.range(-(kernel_w // 2), kernel_w // 2 + 1, dtype=tf.float32)
        kernel_y, kernel_x = tf.meshgrid(kernel_y, kernel_x, indexing='ij')
        
        kernel_grid = tf.stack([kernel_y, kernel_x], axis=-1)
        kernel_grid = tf.reshape(kernel_grid, [-1, 2])
        
        # Broadcast to output spatial dimensions
        kernel_grid = tf.expand_dims(tf.expand_dims(kernel_grid, 0), 0)
        kernel_grid = tf.tile(kernel_grid, [1, height, width, 1, 1])
        
        return kernel_grid
    
    def _bilinear_sample(self, inputs, grid):
        """
        Bilinear sampling from input feature map at grid positions.
        
        Args:
            inputs: Input tensor (batch, h, w, c)
            grid: Sampling grid (batch, h_out, w_out, num_points, 2)
            
        Returns:
            Sampled features
        """
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = inputs.shape[3]
        
        # Flatten grid
        grid_shape = tf.shape(grid)
        grid_flat = tf.reshape(grid, [-1, 2])
        
        # Get y and x coordinates
        y = grid_flat[:, 0]
        x = grid_flat[:, 1]
        
        # Clip to valid range
        y = tf.clip_by_value(y, 0.0, tf.cast(height - 1, tf.float32))
        x = tf.clip_by_value(x, 0.0, tf.cast(width - 1, tf.float32))
        
        # Get corner coordinates for bilinear interpolation
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1
        x0 = tf.cast(tf.floor(x), tf.int32)
        x1 = x0 + 1
        
        y1 = tf.minimum(y1, height - 1)
        x1 = tf.minimum(x1, width - 1)
        
        # Get interpolation weights
        wy1 = y - tf.cast(y0, tf.float32)
        wy0 = 1.0 - wy1
        wx1 = x - tf.cast(x0, tf.float32)
        wx0 = 1.0 - wx1
        
        # Expand dimensions for broadcasting
        wy0 = tf.expand_dims(wy0, -1)
        wy1 = tf.expand_dims(wy1, -1)
        wx0 = tf.expand_dims(wx0, -1)
        wx1 = tf.expand_dims(wx1, -1)
        
        # Flatten inputs for gathering
        inputs_flat = tf.reshape(inputs, [-1, channels])
        
        # Compute batch indices
        batch_idx = tf.range(batch_size)
        batch_idx = tf.reshape(batch_idx, [-1, 1, 1, 1])
        batch_idx = tf.tile(batch_idx, [1, grid_shape[1], grid_shape[2], grid_shape[3]])
        batch_idx = tf.reshape(batch_idx, [-1])
        
        # Compute flat indices
        base = batch_idx * height * width
        idx_00 = base + y0 * width + x0
        idx_01 = base + y0 * width + x1
        idx_10 = base + y1 * width + x0
        idx_11 = base + y1 * width + x1
        
        # Gather values
        v00 = tf.gather(inputs_flat, idx_00)
        v01 = tf.gather(inputs_flat, idx_01)
        v10 = tf.gather(inputs_flat, idx_10)
        v11 = tf.gather(inputs_flat, idx_11)
        
        # Bilinear interpolation
        v0 = wx0 * v00 + wx1 * v01
        v1 = wx0 * v10 + wx1 * v11
        output = wy0 * v0 + wy1 * v1
        
        # Reshape to original grid shape
        output = tf.reshape(
            output,
            [batch_size, grid_shape[1], grid_shape[2], grid_shape[3], channels]
        )
        
        return output
    
    def get_config(self):
        """Return layer configuration."""
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding.lower(),
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'deformable_groups': self.deformable_groups
        }
        base_config = super(DynamicSnakeConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SnakeConvBlock(layers.Layer):
    """
    Snake Convolution Block combining DSC with normalization and activation.
    """
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        **kwargs
    ):
        """
        Initialize Snake Conv Block.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of convolution kernel
            strides: Stride of convolution
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0 = no dropout)
        """
        super(SnakeConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        """Build the layer."""
        # Build layers
        self.snake_conv = DynamicSnakeConv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same'
        )
        
        if self.use_batch_norm:
            self.batch_norm = layers.BatchNormalization()
        
        self.activation_layer = layers.Activation(self.activation_name)
        
        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)
            
        super(SnakeConvBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.snake_conv(inputs, training=training)
        
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
        
        x = self.activation_layer(x)
        
        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        """Return configuration."""
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'activation': self.activation_name,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(SnakeConvBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    # Test Dynamic Snake Convolution
    print("Testing Dynamic Snake Convolution...")
    
    # Create test input
    test_input = tf.random.normal([2, 224, 224, 3])
    
    # Create DSC layer
    dsc_layer = DynamicSnakeConv2D(filters=32, kernel_size=3)
    
    # Forward pass
    output = dsc_layer(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {dsc_layer.count_params():,}")
    
    # Test Snake Conv Block
    print("\nTesting Snake Conv Block...")
    snake_block = SnakeConvBlock(filters=64, kernel_size=3, dropout_rate=0.25)
    output = snake_block(test_input)
    
    print(f"Snake Block output shape: {output.shape}")
    print(f"Snake Block parameters: {snake_block.count_params():,}")
    
    print("\n✓ Dynamic Snake Convolution test passed!")
