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

PHOENIX-v3.1 OPTIMIZED Components (Recommended):
- Unified Liquid-S6-KAN (liquid_s6_kan_unified.py) - Integrated cell with proper Δ modulation
- Optimized Model (model_v3_1_optimized.py) - Production-ready (~1.18M params)
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

# PHOENIX-v3.1 OPTIMIZED components (Recommended for production)
try:
    from models.liquid_s6_kan_unified import (
        UnifiedLiquidS6KANCell,
        LiquidSSMWithDeltaModulation,
        EfficientKANLinear,
        EfficientBSpline
    )
    from models.model_v3_1_optimized import (
        PHOENIXv31Optimized,
        create_phoenix_v31_optimized
    )
except ImportError as e:
    print(f"Warning: Could not import PHOENIX-v3.1 optimized components: {e}")

__all__ = [
    # Legacy
    'create_cnn_model',
    'load_model',
    'create_neurosnake_model',
    # PHOENIX-v3.1 Original
    'SpatialMixer',
    'LiquidS6Block',
    'KANBlock',
    'MultiSpectralConcordanceGating',
    'PHOENIXv31',
    'create_phoenix_v31',
    # PHOENIX-v3.1 Optimized (Recommended)
    'UnifiedLiquidS6KANCell',
    'LiquidSSMWithDeltaModulation',
    'EfficientKANLinear',
    'PHOENIXv31Optimized',
    'create_phoenix_v31_optimized',
]
