"""
NeuroSnake Architecture for Phoenix Protocol.
Hybrid model combining Dynamic Snake Convolutions with MobileViT-v2 blocks.
Prioritizes geometric adaptability and clinical robustness over vanity metrics.

Supports:
- Basic NeuroSnake (DSC + MobileViT)
- NeuroSnake with Coordinate Attention (position-preserving)
"""

import os
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from models.dynamic_snake_conv import DynamicSnakeConv2D, SnakeConvBlock
from models.coordinate_attention import CoordinateAttentionBlock


class MobileViTBlock(layers.Layer):
    """
    Lightweight MobileViT-v2 block for global context at deepest layer.
    Wrapped in large-kernel convolutions to mitigate patchifying artifacts.
    """
    
    def __init__(
        self,
        filters: int,
        num_heads: int = 4,
        mlp_ratio: int = 2,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize MobileViT block.
        
        Args:
            filters: Number of output filters
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout_rate: Dropout rate
        """
        super(MobileViTBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        """Build the layer."""
        # Pre-processing: Large kernel conv for robustness
        self.pre_conv = layers.Conv2D(
            self.filters, 
            kernel_size=5, 
            padding='same',
            name='pre_conv'
        )
        
        # Local representation with depthwise separable conv
        self.dw_conv = layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            name='dw_conv'
        )
        self.bn1 = layers.BatchNormalization(name='bn1')
        
        # Pointwise conv to adjust channels
        self.pw_conv1 = layers.Conv2D(
            self.filters,
            kernel_size=1,
            padding='same',
            name='pw_conv1'
        )
        
        # Transformer block for global context
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.filters // self.num_heads,
            dropout=self.dropout_rate,
            name='mha'
        )
        
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6, name='ln1')
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6, name='ln2')
        
        # MLP
        mlp_hidden = self.filters * self.mlp_ratio
        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden, activation='gelu', name='mlp_fc1'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.filters, name='mlp_fc2'),
            layers.Dropout(self.dropout_rate)
        ], name='mlp')
        
        # Post-processing: Pointwise conv
        self.pw_conv2 = layers.Conv2D(
            self.filters,
            kernel_size=1,
            padding='same',
            name='pw_conv2'
        )
        
        # Post-processing: Large kernel conv for additional robustness
        self.post_conv = layers.Conv2D(
            self.filters,
            kernel_size=5,
            padding='same',
            name='post_conv'
        )
        self.bn2 = layers.BatchNormalization(name='bn2')
        
        super(MobileViTBlock, self).build(input_shape)
        
    def call(self, inputs, training=None):
        """Forward pass."""
        batch = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Pre-process with large kernel for robustness
        x = self.pre_conv(inputs)
        
        # Local representation
        local = self.dw_conv(x)
        local = self.bn1(local, training=training)
        local = self.pw_conv1(local)
        
        # Reshape for transformer: (B, H, W, C) -> (B, H*W, C)
        transformer_input = tf.reshape(local, [batch, height * width, self.filters])
        
        # Apply transformer with residual
        attn_output = self.layer_norm1(transformer_input, training=training)
        attn_output = self.attention(attn_output, attn_output, training=training)
        transformer_output = transformer_input + attn_output
        
        # MLP with residual
        mlp_output = self.layer_norm2(transformer_output, training=training)
        mlp_output = self.mlp(mlp_output, training=training)
        transformer_output = transformer_output + mlp_output
        
        # Reshape back: (B, H*W, C) -> (B, H, W, C)
        transformer_output = tf.reshape(transformer_output, [batch, height, width, self.filters])
        
        # Post-process
        output = self.pw_conv2(transformer_output)
        output = self.post_conv(output)
        output = self.bn2(output, training=training)
        
        # Residual connection
        output = output + inputs
        
        return output
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(MobileViTBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_neurosnake_model(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    num_classes=config.NUM_CLASSES,
    use_mobilevit=True,
    dropout_rate=0.3
):
    """
    Create NeuroSnake model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        use_mobilevit: Whether to include MobileViT block at deepest layer
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Stage 1: Initial processing with standard conv
    x = layers.Conv2D(32, 3, strides=2, padding='same', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_act')(x)
    
    # Stage 2: Snake Conv Block 1 (Stride 4)
    x = SnakeConvBlock(64, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    # Stage 3: Snake Conv Block 2 (Stride 8)
    x = SnakeConvBlock(128, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    # Stage 4: Snake Conv Block 3 (Stride 16)
    x = SnakeConvBlock(256, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block3')(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)
    
    # Stage 5: Deepest layer with optional MobileViT for global context
    x = SnakeConvBlock(512, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block4')(x)
    
    if use_mobilevit:
        x = MobileViTBlock(
            filters=512,
            num_heads=8,
            mlp_ratio=2,
            dropout_rate=dropout_rate,
            name='mobilevit_block'
        )(x)
    
    x = layers.MaxPooling2D(2, name='pool4')(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='fc_bn1')(x)
    x = layers.Dropout(0.5, name='fc_dropout1')(x)
    
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization(name='fc_bn2')(x)
    x = layers.Dropout(0.5, name='fc_dropout2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake')
    
    return model


def create_neurosnake_with_coordinate_attention(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    num_classes=config.NUM_CLASSES,
    dropout_rate=0.3,
    use_mobilevit=True
):
    """
    Create NeuroSnake model with Coordinate Attention (position-preserving).
    
    CRITICAL IMPROVEMENT: Coordinate Attention preserves spatial position
    information, unlike SEVector which destroys it via global average pooling.
    
    Key for medical imaging:
    - Tumor location is diagnostic
    - Boundary delineation requires spatial info
    - Multi-focal lesions need position awareness
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        use_mobilevit: Include MobileViT block for global context
        
    Returns:
        Keras model with Coordinate Attention
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Stage 1: Initial processing with standard conv
    x = layers.Conv2D(32, 3, strides=2, padding='same', name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('relu', name='stem_act')(x)
    
    # Stage 2: Snake Conv + Coordinate Attention (Stride 4)
    x = SnakeConvBlock(64, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block1')(x)
    x = CoordinateAttentionBlock(filters=64, reduction_ratio=8, name='ca1')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    
    # Stage 3: Snake Conv + Coordinate Attention (Stride 8)
    x = SnakeConvBlock(128, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block2')(x)
    x = CoordinateAttentionBlock(filters=128, reduction_ratio=8, name='ca2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    
    # Stage 4: Snake Conv + Coordinate Attention (Stride 16)
    x = SnakeConvBlock(256, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block3')(x)
    x = CoordinateAttentionBlock(filters=256, reduction_ratio=8, name='ca3')(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)
    
    # Stage 5: Deepest layer with optional MobileViT + CA
    x = SnakeConvBlock(512, kernel_size=3, strides=1, dropout_rate=dropout_rate, name='snake_block4')(x)
    
    if use_mobilevit:
        x = MobileViTBlock(
            filters=512,
            num_heads=8,
            mlp_ratio=2,
            dropout_rate=dropout_rate,
            name='mobilevit_block'
        )(x)
    
    # Coordinate Attention at deepest layer for maximum spatial awareness
    x = CoordinateAttentionBlock(filters=512, reduction_ratio=8, name='ca4')(x)
    x = layers.MaxPooling2D(2, name='pool4')(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='fc_bn1')(x)
    x = layers.Dropout(0.5, name='fc_dropout1')(x)
    
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization(name='fc_bn2')(x)
    x = layers.Dropout(0.5, name='fc_dropout2')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='NeuroSnake_CoordinateAttention')
    
    return model


def create_baseline_comparison_model(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    num_classes=config.NUM_CLASSES
):
    """
    Create baseline CNN model for comparison (from arXiv:2504.21188).
    Standard convolutions without snake deformations.
    
    Returns:
        Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1_1')(inputs)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1_2')(x)
    x = layers.MaxPooling2D(2, name='pool1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2_1')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2_2')(x)
    x = layers.MaxPooling2D(2, name='pool2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3_1')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3_2')(x)
    x = layers.MaxPooling2D(2, name='pool3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.Dropout(0.25, name='dropout3')(x)
    
    # Block 4
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv4_1')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv4_2')(x)
    x = layers.MaxPooling2D(2, name='pool4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.Dropout(0.25, name='dropout4')(x)
    
    # Dense layers
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='fc_dropout')(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='BaselineCNN')
    
    return model


if __name__ == "__main__":
    print("="*80)
    print("PHOENIX PROTOCOL: NeuroSnake Model Architecture")
    print("="*80)
    
    # Create NeuroSnake model
    print("\n1. Creating NeuroSnake model...")
    neurosnake = create_neurosnake_model()
    neurosnake.summary()
    
    print(f"\nNeuroSnake Parameters: {neurosnake.count_params():,}")
    
    # Create NeuroSnake with Coordinate Attention
    print("\n" + "="*80)
    print("2. Creating NeuroSnake with Coordinate Attention...")
    neurosnake_ca = create_neurosnake_with_coordinate_attention()
    neurosnake_ca.summary()
    
    print(f"\nNeuroSnake+CA Parameters: {neurosnake_ca.count_params():,}")
    
    # Create baseline model for comparison
    print("\n" + "="*80)
    print("3. Creating Baseline model for comparison...")
    baseline = create_baseline_comparison_model()
    baseline.summary()
    
    print(f"\nBaseline Parameters: {baseline.count_params():,}")
    
    # Test forward pass
    print("\n" + "="*80)
    print("4. Testing forward pass...")
    test_input = tf.random.normal([1, config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS])
    
    neurosnake_output = neurosnake(test_input, training=False)
    neurosnake_ca_output = neurosnake_ca(test_input, training=False)
    baseline_output = baseline(test_input, training=False)
    
    print(f"NeuroSnake output shape: {neurosnake_output.shape}")
    print(f"NeuroSnake+CA output shape: {neurosnake_ca_output.shape}")
    print(f"Baseline output shape: {baseline_output.shape}")
    
    print("\n✓ NeuroSnake architecture created successfully!")
    print("="*80)
