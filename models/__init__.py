"""
Brain Tumor Detection - Models Module

Available Architectures:
- Baseline CNN (cnn_model.py)
- NeuroSnake (neurosnake_model.py)
- Dynamic Snake Convolutions (dynamic_snake_conv.py)
- Coordinate Attention (coordinate_attention.py)

PHOENIX-v3.1 Components:
- SpatialMixer (spatial_mixer.py) - Topology preservation
- Liquid-S6 SSM (liquid_ssm.py) - Adaptive state-space model
- KAN Layer (kan_layer.py) - Kolmogorov-Arnold Network
- Spectral Gating (spectral_gating.py) - Multi-spectral concordance
- PHOENIX-v3.1 Model (model_v3_1.py) - Full hybrid architecture
"""

# Legacy models
from models.cnn_model import create_cnn_model, load_model

# NeuroSnake architecture
try:
    from models.neurosnake_model import create_neurosnake_model
except ImportError:
    pass

# PHOENIX-v3.1 components
try:
    from models.spatial_mixer import SpatialMixer, TopologyPreservingFlatten
    from models.liquid_ssm import S6SelectiveSSM, LiquidS6, LiquidS6Block
    from models.kan_layer import KANLinear, KANBlock, AdaptiveKAN
    from models.spectral_gating import MultiSpectralConcordanceGating
    from models.model_v3_1 import PHOENIXv31, create_phoenix_v31
except ImportError as e:
    print(f"Warning: Could not import PHOENIX-v3.1 components: {e}")

__all__ = [
    'create_cnn_model',
    'load_model',
    'create_neurosnake_model',
    'SpatialMixer',
    'LiquidS6Block',
    'KANBlock',
    'MultiSpectralConcordanceGating',
    'PHOENIXv31',
    'create_phoenix_v31'
]
