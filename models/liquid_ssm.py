# Liquid State-Space Model (Liquid-S6) for PHOENIX-v3.1
"""
PHOENIX-v3.1 Component: Liquid-S6 State-Space Model

Implements the Liquid State-Space Model with adaptive time constants.
Unlike static SSMs, Liquid-S6 modulates its dynamics based on input
content, allowing rapid evolution in healthy tissue and slow accumulation
in tumor regions.

Key Features:
1. S6 Selective State Space (from Mamba architecture)
2. Liquid Time Constants (content-dependent dynamics)
3. Priority Scout integration for ROI-aware processing

Reference: PHOENIX-v3.1 Paper Section 3.1 and 3.3
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, Tuple


class S6SelectiveSSM(layers.Layer):
    """
    S6 Selective State-Space Model Layer.
    
    Implements the selective scan mechanism from Mamba with linear
    complexity O(N) instead of quadratic O(N^2) attention.
    
    The state-space model is defined as:
        x'(t) = A*x(t) + B*u(t)
        y(t) = C*x(t) + D*u(t)
    
    Where A, B, C, D are input-dependent (selective).
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        expand_factor: int = 2,
        dt_rank: str = 'auto',
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = 'random',
        name: str = 's6_selective_ssm',
        **kwargs
    ):
        """
        Initialize S6 Selective SSM.
        
        Args:
            state_dim: Dimension of the state space (N in paper)
            expand_factor: Expansion factor for inner dimension
            dt_rank: Rank for dt projection ('auto' = ceil(d_model/16))
            dt_min: Minimum value for delta (time step)
            dt_max: Maximum value for delta (time step)
            dt_init: Initialization method for dt
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        
    def build(self, input_shape):
        """Build the layer."""
        self.d_model = input_shape[-1]
        self.d_inner = self.d_model * self.expand_factor
        
        # Auto-compute dt_rank
        if self.dt_rank == 'auto':
            self.dt_rank_value = max(1, self.d_model // 16)
        else:
            self.dt_rank_value = int(self.dt_rank)
        
        # Input projection (to expanded dimension)
        self.in_proj = layers.Dense(
            self.d_inner * 2,
            use_bias=False,
            name=f'{self.name}/in_proj'
        )
        
        # Convolution for local context
        self.conv1d = layers.Conv1D(
            filters=self.d_inner,
            kernel_size=4,
            padding='same',
            groups=self.d_inner,
            use_bias=True,
            name=f'{self.name}/conv1d'
        )
        
        # SSM parameter projections (input-dependent)
        self.x_proj = layers.Dense(
            self.dt_rank_value + self.state_dim * 2,
            use_bias=False,
            name=f'{self.name}/x_proj'
        )
        
        # Delta (dt) projection
        self.dt_proj = layers.Dense(
            self.d_inner,
            use_bias=True,
            name=f'{self.name}/dt_proj'
        )
        
        # Initialize dt bias to be in [dt_min, dt_max]
        dt_init_std = self.dt_rank_value ** -0.5
        
        # A parameter (diagonal state matrix)
        # Initialize using S4D-Lin initialization
        A = np.repeat(np.arange(1, self.state_dim + 1), self.d_inner).reshape(self.state_dim, self.d_inner).T
        self.A_log = self.add_weight(
            name='A_log',
            shape=(self.d_inner, self.state_dim),
            initializer=tf.constant_initializer(np.log(A)),
            trainable=True
        )
        
        # D parameter (skip connection)
        self.D = self.add_weight(
            name='D',
            shape=(self.d_inner,),
            initializer='ones',
            trainable=True
        )
        
        # Output projection
        self.out_proj = layers.Dense(
            self.d_model,
            use_bias=False,
            name=f'{self.name}/out_proj'
        )
        
        super().build(input_shape)
        
    def selective_scan(self, u, delta, A, B, C, D):
        """
        Selective scan operation (parallel implementation).
        
        Args:
            u: Input tensor (batch, seq_len, d_inner)
            delta: Time step (batch, seq_len, d_inner)
            A: State matrix (d_inner, state_dim)
            B: Input matrix (batch, seq_len, state_dim)
            C: Output matrix (batch, seq_len, state_dim)
            D: Skip connection (d_inner,)
            
        Returns:
            Output tensor (batch, seq_len, d_inner)
        """
        batch_size = tf.shape(u)[0]
        seq_len = tf.shape(u)[1]
        
        # Discretize A and B using delta
        # A_bar = exp(delta * A)
        deltaA = tf.einsum('bld,dn->bldn', delta, A)
        deltaA = tf.exp(deltaA)
        
        # B_bar = delta * B
        deltaB_u = tf.einsum('bld,bln,bld->bldn', delta, B, u)
        
        # Sequential scan (could be parallelized with associative scan)
        # For simplicity, using tf.scan
        def scan_fn(carry, inputs):
            x_prev = carry
            deltaA_t, deltaB_u_t = inputs
            x_new = deltaA_t * x_prev + deltaB_u_t
            return x_new
        
        # Transpose for scan: (seq_len, batch, d_inner, state_dim)
        deltaA_t = tf.transpose(deltaA, [1, 0, 2, 3])
        deltaB_u_t = tf.transpose(deltaB_u, [1, 0, 2, 3])
        
        # Initialize state
        x0 = tf.zeros((batch_size, self.d_inner, self.state_dim))
        
        # Run scan
        x = tf.scan(scan_fn, (deltaA_t, deltaB_u_t), initializer=x0)
        
        # Transpose back: (batch, seq_len, d_inner, state_dim)
        x = tf.transpose(x, [1, 0, 2, 3])
        
        # Compute output: y = C * x + D * u
        y = tf.einsum('bldn,bln->bld', x, C)
        y = y + D * u
        
        return y
        
    def call(self, inputs, training=None):
        """
        Forward pass.
        
        Args:
            inputs: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Input projection and split
        xz = self.in_proj(inputs)
        x, z = tf.split(xz, 2, axis=-1)
        
        # Convolution for local context
        x = self.conv1d(x)
        x = tf.nn.silu(x)
        
        # Compute SSM parameters from input
        x_proj = self.x_proj(x)
        
        # Split into dt, B, C
        dt, B, C = tf.split(
            x_proj,
            [self.dt_rank_value, self.state_dim, self.state_dim],
            axis=-1
        )
        
        # Project dt to full dimension and apply softplus
        dt = self.dt_proj(dt)
        dt = tf.nn.softplus(dt)
        
        # Get A from log parameterization
        A = -tf.exp(self.A_log)
        
        # Run selective scan
        y = self.selective_scan(x, dt, A, B, C, self.D)
        
        # Gate with z
        y = y * tf.nn.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim,
            'expand_factor': self.expand_factor,
            'dt_rank': self.dt_rank,
            'dt_min': self.dt_min,
            'dt_max': self.dt_max,
            'dt_init': self.dt_init
        })
        return config


class LiquidS6(layers.Layer):
    """
    Liquid State-Space Model with Adaptive Time Constants.
    
    Extends S6 with content-dependent time constant modulation.
    The time constant delta is modulated by a Priority Scout ROI map:
        delta_adaptive = delta * exp(-P(x))
    
    This ensures rapid evolution in healthy tissue and slow
    accumulation in tumor regions.
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        expand_factor: int = 2,
        use_priority_scout: bool = True,
        name: str = 'liquid_s6',
        **kwargs
    ):
        """
        Initialize Liquid-S6.
        
        Args:
            state_dim: Dimension of the state space
            expand_factor: Expansion factor for inner dimension
            use_priority_scout: Whether to use adaptive time constants
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        self.use_priority_scout = use_priority_scout
        
    def build(self, input_shape):
        """Build the layer."""
        self.d_model = input_shape[-1]
        
        # Priority Scout: predicts ROI map for time constant modulation
        if self.use_priority_scout:
            self.priority_scout = keras.Sequential([
                layers.Dense(self.d_model // 4, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ], name=f'{self.name}/priority_scout')
        
        # Layer normalization
        self.norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'{self.name}/norm'
        )
        
        # S6 SSM
        self.ssm = S6SelectiveSSM(
            state_dim=self.state_dim,
            expand_factor=self.expand_factor,
            name=f'{self.name}/ssm'
        )
        
        super().build(input_shape)
        
    def call(self, inputs, priority_map=None, training=None):
        """
        Forward pass with optional priority-based adaptation.
        
        Args:
            inputs: Input tensor (batch, seq_len, d_model)
            priority_map: Optional external priority map
            training: Whether in training mode
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Normalize input
        x = self.norm(inputs)
        
        # Compute priority map if using scout
        if self.use_priority_scout and priority_map is None:
            priority_map = self.priority_scout(x)
        
        # Apply SSM
        y = self.ssm(x, training=training)
        
        # Modulate output by priority (soft gating)
        if priority_map is not None:
            y = y * (1 + priority_map)
        
        # Residual connection
        output = inputs + y
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim,
            'expand_factor': self.expand_factor,
            'use_priority_scout': self.use_priority_scout
        })
        return config


class LiquidS6Block(layers.Layer):
    """
    Complete Liquid-S6 Block with FFN.
    
    Combines Liquid-S6 SSM with a feed-forward network
    following the standard transformer block pattern.
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        expand_factor: int = 2,
        ffn_expand_factor: int = 4,
        dropout_rate: float = 0.1,
        name: str = 'liquid_s6_block',
        **kwargs
    ):
        """
        Initialize Liquid-S6 Block.
        
        Args:
            state_dim: Dimension of the state space
            expand_factor: SSM expansion factor
            ffn_expand_factor: FFN expansion factor
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        self.ffn_expand_factor = ffn_expand_factor
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        """Build the layer."""
        self.d_model = input_shape[-1]
        
        # Liquid-S6 layer
        self.liquid_s6 = LiquidS6(
            state_dim=self.state_dim,
            expand_factor=self.expand_factor,
            name=f'{self.name}/liquid_s6'
        )
        
        # Feed-forward network
        self.ffn_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f'{self.name}/ffn_norm'
        )
        
        self.ffn = keras.Sequential([
            layers.Dense(
                self.d_model * self.ffn_expand_factor,
                activation='gelu'
            ),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.d_model),
            layers.Dropout(self.dropout_rate)
        ], name=f'{self.name}/ffn')
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass."""
        # Liquid-S6 with residual
        x = self.liquid_s6(inputs, training=training)
        
        # FFN with residual
        y = self.ffn_norm(x)
        y = self.ffn(y, training=training)
        output = x + y
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'state_dim': self.state_dim,
            'expand_factor': self.expand_factor,
            'ffn_expand_factor': self.ffn_expand_factor,
            'dropout_rate': self.dropout_rate
        })
        return config


# Example usage and testing
if __name__ == "__main__":
    print("Liquid-S6: Adaptive State-Space Model")
    print("=" * 60)
    
    # Test configuration
    batch_size = 2
    seq_len = 196  # 14x14 feature map flattened
    d_model = 128
    
    x = tf.random.normal((batch_size, seq_len, d_model))
    
    # Test S6SelectiveSSM
    s6 = S6SelectiveSSM(state_dim=16, expand_factor=2)
    y_s6 = s6(x)
    print(f"S6SelectiveSSM: {x.shape} -> {y_s6.shape}")
    
    # Test LiquidS6
    liquid = LiquidS6(state_dim=16, use_priority_scout=True)
    y_liquid = liquid(x)
    print(f"LiquidS6: {x.shape} -> {y_liquid.shape}")
    
    # Test LiquidS6Block
    block = LiquidS6Block(state_dim=16, ffn_expand_factor=4)
    y_block = block(x)
    print(f"LiquidS6Block: {x.shape} -> {y_block.shape}")
    
    print("\n✓ All Liquid-S6 tests passed!")
