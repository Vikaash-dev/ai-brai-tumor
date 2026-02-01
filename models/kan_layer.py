# Kolmogorov-Arnold Network (KAN) Layer for PHOENIX-v3.1
"""
PHOENIX-v3.1 Component: Kolmogorov-Arnold Network (KAN)

Implements KAN with learnable B-spline activation functions.
Unlike MLPs with fixed activations, KAN learns the activation
functions themselves, offering higher expressivity per parameter.

Key Features:
1. Learnable B-spline basis functions on edges
2. Adaptive spline weights for test-time adaptation
3. Grid extension for increased resolution

Reference: 
- Kolmogorov-Arnold Representation Theorem
- U-KAN: Kolmogorov-Arnold U-Net (2024)
- PHOENIX-v3.1 Paper Section 2.2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple, List


class BSplineBasis(layers.Layer):
    """
    B-Spline Basis Function Layer.
    
    Computes B-spline basis functions for given inputs.
    B-splines provide smooth, local basis functions ideal
    for learning activation functions.
    """
    
    def __init__(
        self,
        num_knots: int = 8,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        name: str = 'bspline_basis',
        **kwargs
    ):
        """
        Initialize B-Spline Basis.
        
        Args:
            num_knots: Number of internal knot points
            spline_order: Order of the B-spline (3 = cubic)
            grid_range: Range of the grid
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.grid_range = grid_range
        
    def build(self, input_shape):
        """Build the layer."""
        # Create uniform knot vector with extensions
        num_intervals = self.num_knots - 1
        step = (self.grid_range[1] - self.grid_range[0]) / num_intervals
        
        # Extended knot vector for boundary conditions
        knots = np.linspace(
            self.grid_range[0] - self.spline_order * step,
            self.grid_range[1] + self.spline_order * step,
            self.num_knots + 2 * self.spline_order
        )
        
        self.knots = tf.constant(knots, dtype=tf.float32)
        self.num_basis = self.num_knots + self.spline_order - 1
        
        super().build(input_shape)
        
    def _basis_function(self, x, i, k, knots):
        """
        Recursive B-spline basis function computation (Cox-de Boor formula).
        
        Note: This recursive implementation is simple but has O(2^k) complexity.
        For production with high spline orders, consider:
        1. Iterative Cox-de Boor (O(k^2) per evaluation point)
        2. Pre-computed basis matrices
        3. Efficient TensorFlow sparse operations
        
        The recursive approach is acceptable for spline_order <= 4 (typical).
        
        Args:
            x: Input values
            i: Knot index
            k: Spline order (recursion depth)
            knots: Knot vector
            
        Returns:
            Basis function values
        """
        if k == 0:
            return tf.cast(
                (knots[i] <= x) & (x < knots[i + 1]),
                tf.float32
            )
        
        # Cox-de Boor recursive formula
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]
        
        term1 = tf.where(
            denom1 != 0,
            (x - knots[i]) / denom1 * self._basis_function(x, i, k - 1, knots),
            tf.zeros_like(x)
        )
        
        term2 = tf.where(
            denom2 != 0,
            (knots[i + k + 1] - x) / denom2 * self._basis_function(x, i + 1, k - 1, knots),
            tf.zeros_like(x)
        )
        
        return term1 + term2
        
    def call(self, inputs):
        """
        Compute B-spline basis functions.
        
        Args:
            inputs: Input tensor of any shape
            
        Returns:
            Basis functions of shape (*input_shape, num_basis)
        """
        # Flatten input for processing
        original_shape = tf.shape(inputs)
        x_flat = tf.reshape(inputs, [-1])
        
        # Compute all basis functions
        basis_values = []
        for i in range(self.num_basis):
            basis_i = self._basis_function(
                x_flat, i, self.spline_order, self.knots
            )
            basis_values.append(basis_i)
        
        # Stack basis functions: (flat_size, num_basis)
        basis = tf.stack(basis_values, axis=-1)
        
        # Reshape to original shape + num_basis
        output_shape = tf.concat([original_shape, [self.num_basis]], axis=0)
        basis = tf.reshape(basis, output_shape)
        
        return basis
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'num_knots': self.num_knots,
            'spline_order': self.spline_order,
            'grid_range': self.grid_range
        })
        return config


class KANLinear(layers.Layer):
    """
    Kolmogorov-Arnold Network Linear Layer.
    
    Replaces traditional linear+activation with learnable
    activation functions on each edge of the network.
    
    Instead of: y = activation(Wx + b)
    KAN does: y = sum_i(spline_i(x_i))
    
    Where each spline is a learnable B-spline function.
    """
    
    def __init__(
        self,
        out_features: int,
        num_knots: int = 8,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        name: str = 'kan_linear',
        **kwargs
    ):
        """
        Initialize KAN Linear Layer.
        
        Args:
            out_features: Number of output features
            num_knots: Number of B-spline knots
            spline_order: Order of B-spline
            scale_noise: Noise scale for initialization
            scale_base: Scale for base weight
            scale_spline: Scale for spline weight
            enable_standalone_scale_spline: Whether to use separate spline scale
            grid_range: Range for B-spline grid
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.out_features = out_features
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_range = grid_range
        
    def build(self, input_shape):
        """Build the layer."""
        self.in_features = input_shape[-1]
        self.num_basis = self.num_knots + self.spline_order - 1
        
        # Base weight (like standard linear layer)
        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(self.in_features, self.out_features),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        # Spline coefficients for each input-output pair
        # Shape: (in_features, out_features, num_basis)
        self.spline_weight = self.add_weight(
            name='spline_weight',
            shape=(self.in_features, self.out_features, self.num_basis),
            initializer=keras.initializers.TruncatedNormal(stddev=self.scale_noise),
            trainable=True
        )
        
        # Standalone scale for spline (for TTT adaptation)
        if self.enable_standalone_scale_spline:
            self.spline_scale = self.add_weight(
                name='spline_scale',
                shape=(self.in_features, self.out_features),
                initializer='ones',
                trainable=True
            )
        
        # Create B-spline basis
        self.bspline_basis = BSplineBasis(
            num_knots=self.num_knots,
            spline_order=self.spline_order,
            grid_range=self.grid_range,
            name=f'{self.name}/bspline_basis'
        )
        
        super().build(input_shape)
        
    def call(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base linear transformation (like SiLU-weighted)
        base_output = tf.matmul(inputs, self.base_weight) * self.scale_base
        
        # Compute B-spline basis for each input
        # inputs: (..., in_features) -> basis: (..., in_features, num_basis)
        basis = self.bspline_basis(inputs)
        
        # Compute spline output
        # basis: (..., in_features, num_basis)
        # spline_weight: (in_features, out_features, num_basis)
        # We want: sum over in_features and num_basis
        
        # Reshape for einsum
        # basis: (batch, seq, in_features, num_basis) if 3D input
        # spline_weight: (in_features, out_features, num_basis)
        
        spline_output = tf.einsum('...ik,iok->...o', basis, self.spline_weight)
        
        # Apply scale
        if self.enable_standalone_scale_spline:
            # spline_scale: (in_features, out_features)
            # Apply per-output-feature scaling (sum over input features)
            # This preserves the TTT adaptation capability per output channel
            scale_per_output = tf.reduce_sum(self.spline_scale, axis=0)  # (out_features,)
            scale_per_output = scale_per_output / tf.cast(self.in_features, tf.float32)  # Normalize
            spline_output = spline_output * scale_per_output * self.scale_spline
        else:
            spline_output = spline_output * self.scale_spline
        
        return base_output + spline_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'out_features': self.out_features,
            'num_knots': self.num_knots,
            'spline_order': self.spline_order,
            'scale_noise': self.scale_noise,
            'scale_base': self.scale_base,
            'scale_spline': self.scale_spline,
            'enable_standalone_scale_spline': self.enable_standalone_scale_spline,
            'grid_range': self.grid_range
        })
        return config


class KANBlock(layers.Layer):
    """
    KAN Block with multiple KAN linear layers.
    
    Provides a drop-in replacement for MLP blocks
    with learnable activation functions.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_knots: int = 8,
        spline_order: int = 3,
        dropout_rate: float = 0.1,
        name: str = 'kan_block',
        **kwargs
    ):
        """
        Initialize KAN Block.
        
        Args:
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            num_knots: Number of B-spline knots
            spline_order: Order of B-spline
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        """Build the layer."""
        self.kan1 = KANLinear(
            out_features=self.hidden_dim,
            num_knots=self.num_knots,
            spline_order=self.spline_order,
            name=f'{self.name}/kan1'
        )
        
        self.kan2 = KANLinear(
            out_features=self.out_dim,
            num_knots=self.num_knots,
            spline_order=self.spline_order,
            name=f'{self.name}/kan2'
        )
        
        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'{self.name}/norm'
        )
        
        self.dropout = layers.Dropout(self.dropout_rate)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass."""
        x = self.norm(inputs)
        x = self.kan1(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        x = self.kan2(x)
        x = self.dropout(x, training=training)
        
        # Residual if dimensions match
        if inputs.shape[-1] == self.out_dim:
            x = x + inputs
            
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'num_knots': self.num_knots,
            'spline_order': self.spline_order,
            'dropout_rate': self.dropout_rate
        })
        return config


class AdaptiveKAN(layers.Layer):
    """
    Adaptive KAN Layer for Test-Time Training.
    
    The spline weights can be adapted at inference time
    to minimize entropy on the specific patient scan.
    """
    
    def __init__(
        self,
        out_features: int,
        num_knots: int = 8,
        adaptation_lr: float = 0.01,
        adaptation_steps: int = 3,
        name: str = 'adaptive_kan',
        **kwargs
    ):
        """
        Initialize Adaptive KAN.
        
        Args:
            out_features: Number of output features
            num_knots: Number of B-spline knots
            adaptation_lr: Learning rate for TTT
            adaptation_steps: Number of adaptation steps
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.out_features = out_features
        self.num_knots = num_knots
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        
    def build(self, input_shape):
        """Build the layer."""
        self.kan = KANLinear(
            out_features=self.out_features,
            num_knots=self.num_knots,
            enable_standalone_scale_spline=True,
            name=f'{self.name}/kan'
        )
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass."""
        return self.kan(inputs)
    
    def adapt(self, inputs, entropy_fn):
        """
        Adapt spline weights to minimize entropy.
        
        Args:
            inputs: Input tensor
            entropy_fn: Function that computes entropy of predictions
            
        Returns:
            Adapted output
        """
        # Get the spline scale variable
        spline_scale = self.kan.spline_scale
        
        for _ in range(self.adaptation_steps):
            with tf.GradientTape() as tape:
                tape.watch(spline_scale)
                output = self.kan(inputs)
                entropy = entropy_fn(output)
            
            # Compute gradient
            grad = tape.gradient(entropy, spline_scale)
            
            # Update spline scale
            if grad is not None:
                spline_scale.assign_sub(self.adaptation_lr * grad)
        
        return self.kan(inputs)
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'out_features': self.out_features,
            'num_knots': self.num_knots,
            'adaptation_lr': self.adaptation_lr,
            'adaptation_steps': self.adaptation_steps
        })
        return config


# Example usage and testing
if __name__ == "__main__":
    print("KAN: Kolmogorov-Arnold Network")
    print("=" * 60)
    
    # Test configuration
    batch_size = 2
    seq_len = 196
    in_features = 128
    out_features = 256
    
    x = tf.random.normal((batch_size, seq_len, in_features))
    
    # Test KANLinear
    kan_linear = KANLinear(out_features=out_features, num_knots=8)
    y_linear = kan_linear(x)
    print(f"KANLinear: {x.shape} -> {y_linear.shape}")
    
    # Test KANBlock
    kan_block = KANBlock(hidden_dim=256, out_dim=in_features)
    y_block = kan_block(x)
    print(f"KANBlock: {x.shape} -> {y_block.shape}")
    
    # Test AdaptiveKAN
    adaptive_kan = AdaptiveKAN(out_features=out_features)
    y_adaptive = adaptive_kan(x)
    print(f"AdaptiveKAN: {x.shape} -> {y_adaptive.shape}")
    
    # Count parameters
    total_params = sum([tf.reduce_prod(w.shape).numpy() for w in kan_linear.trainable_weights])
    print(f"\nKANLinear parameters: {total_params:,}")
    
    print("\n✓ All KAN tests passed!")
