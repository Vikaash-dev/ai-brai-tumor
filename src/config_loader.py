"""
Configuration Loader for PHOENIX-v3.1

This module provides utilities to load and manage configuration from YAML files,
replacing hardcoded values throughout the codebase.

Reference: Appendix A.2 - Medium Issues (Config Hardcoding)
"""

import os
import yaml
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class InputConfig:
    """Input configuration."""
    width: int = 224
    height: int = 224
    channels: int = 3
    use_true_2_5d: bool = True
    slice_context: int = 3


@dataclass
class StemConfig:
    """Stem layer configuration."""
    filters: int = 32
    kernel_size: int = 7
    stride: int = 2
    use_batch_norm: bool = True


@dataclass
class ConvStageConfig:
    """Convolutional stage configuration."""
    filters: int = 64
    num_blocks: int = 2
    stride: int = 2
    use_residual: bool = True


@dataclass
class SSMStageConfig:
    """SSM stage configuration."""
    filters: int = 128
    state_dim: int = 16
    kernel_size: int = 4
    num_cells: int = 2
    use_kan: bool = True


@dataclass
class HybridPyramidConfig:
    """Hybrid pyramid architecture configuration."""
    stem: StemConfig = field(default_factory=StemConfig)
    conv_stages: Dict[str, ConvStageConfig] = field(default_factory=dict)
    ssm_stages: Dict[str, SSMStageConfig] = field(default_factory=dict)


@dataclass
class SpatialMixerConfig:
    """SpatialMixer configuration."""
    kernel_size: int = 3
    use_depthwise: bool = True
    activation: str = "swish"
    use_residual: bool = True


@dataclass
class LiquidSSMConfig:
    """Liquid-S6 SSM configuration."""
    state_dim: int = 16
    dt_rank: Union[str, int] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    a_init: str = "s4d_lin"
    use_liquid_modulation: bool = True
    tau_min: float = 0.1
    tau_max: float = 10.0


@dataclass
class KANConfig:
    """KAN layer configuration."""
    grid_size: int = 5
    spline_order: int = 3
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone: bool = True
    grid_eps: float = 0.02
    grid_range: tuple = (-1, 1)
    ttt_enabled: bool = True
    ttt_spline_only: bool = True


@dataclass
class SpectralGatingConfig:
    """Multi-Spectral Concordance Gating configuration."""
    enabled: bool = True
    fft_norm: str = "ortho"
    concordance_threshold: float = 0.5
    num_modalities: int = 4
    fusion_method: str = "learned"


@dataclass
class PriorityScoutConfig:
    """Priority Scout configuration."""
    enabled: bool = True
    kernel_size: int = 7
    activation: str = "sigmoid"
    modulation_type: str = "delta"
    modulation_strength: float = 1.0


@dataclass
class TTTConfig:
    """Test-Time Training configuration."""
    enabled: bool = True
    mode: str = "symbolic_mirror"
    target: str = "spline_only"
    entropy_threshold: float = 0.3
    learning_rate: float = 0.001
    max_steps: int = 5
    early_stop_delta: float = 0.001


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adan"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    beta1: float = 0.98
    beta2: float = 0.92
    beta3: float = 0.99
    eps: float = 1e-8


@dataclass
class LRScheduleConfig:
    """Learning rate schedule configuration."""
    type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-7


@dataclass
class LossConfig:
    """Loss function configuration."""
    name: str = "focal"
    alpha: float = 0.25
    gamma: float = 2.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5


@dataclass
class InputValidationConfig:
    """Input validation configuration."""
    enabled: bool = True
    max_image_width: int = 512
    max_image_height: int = 512
    max_batch_size: int = 64
    check_nan: bool = True
    check_inf: bool = True
    check_range: bool = True
    value_range: tuple = (0.0, 1.0)
    expected_dtype: str = "float32"
    max_tensor_size_mb: float = 100.0


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    mixed_precision: bool = True
    xla_compilation: bool = False
    gradient_checkpointing: bool = False
    max_gpu_memory_mb: Optional[int] = None
    multi_gpu: bool = False
    strategy: str = "mirrored"


@dataclass
class PathsConfig:
    """Paths configuration."""
    base_dir: str = "."
    data_dir: str = "data"
    models_dir: str = "models/saved_models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    checkpoints_dir: str = "checkpoints"


@dataclass
class PhoenixConfig:
    """Complete PHOENIX-v3.1 configuration."""
    model_name: str = "phoenix_v3_1"
    model_version: str = "3.1.0"
    num_classes: int = 2
    class_names: tuple = ("no_tumor", "tumor")
    
    input: InputConfig = field(default_factory=InputConfig)
    hybrid_pyramid: HybridPyramidConfig = field(default_factory=HybridPyramidConfig)
    spatial_mixer: SpatialMixerConfig = field(default_factory=SpatialMixerConfig)
    liquid_ssm: LiquidSSMConfig = field(default_factory=LiquidSSMConfig)
    kan: KANConfig = field(default_factory=KANConfig)
    spectral_gating: SpectralGatingConfig = field(default_factory=SpectralGatingConfig)
    priority_scout: PriorityScoutConfig = field(default_factory=PriorityScoutConfig)
    ttt: TTTConfig = field(default_factory=TTTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    input_validation: InputValidationConfig = field(default_factory=InputValidationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# =============================================================================
# Configuration Loader
# =============================================================================

class ConfigLoader:
    """
    Load and manage PHOENIX-v3.1 configuration.
    
    Features:
    - Load from YAML file
    - Merge with defaults
    - Environment variable overrides
    - Validation
    
    Example:
        >>> loader = ConfigLoader("config.yaml")
        >>> config = loader.load()
        >>> print(config.training.batch_size)
    """
    
    DEFAULT_CONFIG_PATH = "config.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config_dict: Dict[str, Any] = {}
        self._config: Optional[PhoenixConfig] = None
    
    def load(self) -> PhoenixConfig:
        """
        Load configuration from YAML file.
        
        Returns:
            PhoenixConfig object
        """
        # Load YAML file if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self._config_dict = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._config_dict = {}
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Build config object
        self._config = self._build_config()
        
        return self._config
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            "PHOENIX_BATCH_SIZE": ("training", "batch_size", int),
            "PHOENIX_EPOCHS": ("training", "epochs", int),
            "PHOENIX_LEARNING_RATE": ("training", "optimizer", "learning_rate", float),
            "PHOENIX_IMG_WIDTH": ("model", "input", "width", int),
            "PHOENIX_IMG_HEIGHT": ("model", "input", "height", int),
            "PHOENIX_NUM_CLASSES": ("model", "classification", "num_classes", int),
            "PHOENIX_MIXED_PRECISION": ("hardware", "mixed_precision", bool),
        }
        
        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                *keys, dtype = path
                self._set_nested(self._config_dict, keys, dtype(value))
                logger.info(f"Override from env: {env_var}={value}")
    
    def _set_nested(self, d: dict, keys: list, value: Any) -> None:
        """Set a nested dictionary value."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def _get_nested(self, d: dict, keys: list, default: Any = None) -> Any:
        """Get a nested dictionary value."""
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, default)
            else:
                return default
        return d
    
    def _build_config(self) -> PhoenixConfig:
        """Build PhoenixConfig from dictionary."""
        # Helper to get config section
        def get(section: str, key: str = None, default: Any = None):
            if key:
                return self._get_nested(self._config_dict, [section, key], default)
            return self._config_dict.get(section, default or {})
        
        # Build input config
        input_cfg = get("model", "input", {})
        input_config = InputConfig(
            width=input_cfg.get("width", 224),
            height=input_cfg.get("height", 224),
            channels=input_cfg.get("channels", 3),
            use_true_2_5d=input_cfg.get("use_true_2_5d", True),
            slice_context=input_cfg.get("slice_context", 3)
        )
        
        # Build training config
        training_cfg = get("training", default={})
        opt_cfg = training_cfg.get("optimizer", {})
        lr_cfg = training_cfg.get("lr_schedule", {})
        loss_cfg = training_cfg.get("loss", {})
        
        training_config = TrainingConfig(
            batch_size=training_cfg.get("batch_size", 32),
            epochs=training_cfg.get("epochs", 100),
            validation_split=training_cfg.get("validation_split", 0.2),
            optimizer=OptimizerConfig(
                name=opt_cfg.get("name", "adan"),
                learning_rate=opt_cfg.get("learning_rate", 0.001),
                weight_decay=opt_cfg.get("weight_decay", 0.0001),
                beta1=opt_cfg.get("beta1", 0.98),
                beta2=opt_cfg.get("beta2", 0.92),
                beta3=opt_cfg.get("beta3", 0.99),
                eps=opt_cfg.get("eps", 1e-8)
            ),
            lr_schedule=LRScheduleConfig(
                type=lr_cfg.get("type", "cosine"),
                warmup_epochs=lr_cfg.get("warmup_epochs", 5),
                min_lr=lr_cfg.get("min_lr", 1e-7)
            ),
            loss=LossConfig(
                name=loss_cfg.get("name", "focal"),
                alpha=loss_cfg.get("alpha", 0.25),
                gamma=loss_cfg.get("gamma", 2.0)
            ),
            dropout_rate=training_cfg.get("dropout_rate", 0.1),
            label_smoothing=training_cfg.get("label_smoothing", 0.1)
        )
        
        # Build input validation config
        val_cfg = get("input_validation", default={})
        input_validation_config = InputValidationConfig(
            enabled=val_cfg.get("enabled", True),
            max_image_width=val_cfg.get("max_image_width", 512),
            max_image_height=val_cfg.get("max_image_height", 512),
            max_batch_size=val_cfg.get("max_batch_size", 64),
            check_nan=val_cfg.get("check_nan", True),
            check_inf=val_cfg.get("check_inf", True)
        )
        
        # Build hardware config
        hw_cfg = get("hardware", default={})
        hardware_config = HardwareConfig(
            mixed_precision=hw_cfg.get("mixed_precision", True),
            xla_compilation=hw_cfg.get("xla_compilation", False),
            gradient_checkpointing=hw_cfg.get("gradient_checkpointing", False),
            multi_gpu=hw_cfg.get("multi_gpu", False),
            strategy=hw_cfg.get("strategy", "mirrored")
        )
        
        # Build paths config
        paths_cfg = get("paths", default={})
        paths_config = PathsConfig(
            base_dir=paths_cfg.get("base_dir", "."),
            data_dir=paths_cfg.get("data_dir", "data"),
            models_dir=paths_cfg.get("models_dir", "models/saved_models"),
            results_dir=paths_cfg.get("results_dir", "results"),
            logs_dir=paths_cfg.get("logs_dir", "logs"),
            checkpoints_dir=paths_cfg.get("checkpoints_dir", "checkpoints")
        )
        
        # Build SSM config
        ssm_cfg = get("liquid_ssm", default={})
        liquid_ssm_config = LiquidSSMConfig(
            state_dim=ssm_cfg.get("state_dim", 16),
            dt_rank=ssm_cfg.get("dt_rank", "auto"),
            dt_min=ssm_cfg.get("dt_min", 0.001),
            dt_max=ssm_cfg.get("dt_max", 0.1),
            use_liquid_modulation=ssm_cfg.get("use_liquid_modulation", True)
        )
        
        # Build KAN config
        kan_cfg = get("kan", default={})
        kan_config = KANConfig(
            grid_size=kan_cfg.get("grid_size", 5),
            spline_order=kan_cfg.get("spline_order", 3),
            ttt_enabled=kan_cfg.get("ttt_enabled", True),
            ttt_spline_only=kan_cfg.get("ttt_spline_only", True)
        )
        
        # Build TTT config
        ttt_cfg = get("ttt_adapter", default={})
        ttt_config = TTTConfig(
            enabled=ttt_cfg.get("enabled", True),
            mode=ttt_cfg.get("mode", "symbolic_mirror"),
            target=ttt_cfg.get("target", "spline_only"),
            entropy_threshold=ttt_cfg.get("entropy_threshold", 0.3),
            learning_rate=ttt_cfg.get("learning_rate", 0.001),
            max_steps=ttt_cfg.get("max_steps", 5)
        )
        
        # Get classification info
        class_cfg = get("model", "classification", {})
        num_classes = class_cfg.get("num_classes", 2)
        class_names = tuple(class_cfg.get("class_names", ["no_tumor", "tumor"]))
        
        return PhoenixConfig(
            model_name="phoenix_v3_1",
            model_version="3.1.0",
            num_classes=num_classes,
            class_names=class_names,
            input=input_config,
            training=training_config,
            input_validation=input_validation_config,
            hardware=hardware_config,
            paths=paths_config,
            liquid_ssm=liquid_ssm_config,
            kan=kan_config,
            ttt=ttt_config
        )
    
    def save(self, config: PhoenixConfig, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object
            path: Output path
        """
        # Convert dataclass to dict
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return [to_dict(v) for v in obj]
            else:
                return obj
        
        config_dict = to_dict(config)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {path}")
    
    @property
    def config(self) -> PhoenixConfig:
        """Get loaded configuration."""
        if self._config is None:
            self._config = self.load()
        return self._config


# =============================================================================
# Global Configuration Instance
# =============================================================================

_global_config: Optional[PhoenixConfig] = None


def get_config(config_path: Optional[str] = None) -> PhoenixConfig:
    """
    Get global configuration instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        PhoenixConfig instance
    """
    global _global_config
    
    if _global_config is None or config_path is not None:
        loader = ConfigLoader(config_path)
        _global_config = loader.load()
    
    return _global_config


def reset_config() -> None:
    """Reset global configuration."""
    global _global_config
    _global_config = None


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Load configuration
    config = get_config("config.yaml")
    
    print("PHOENIX-v3.1 Configuration")
    print("=" * 50)
    print(f"Model: {config.model_name} v{config.model_version}")
    print(f"Classes: {config.num_classes} ({config.class_names})")
    print(f"\nInput: {config.input.width}x{config.input.height}x{config.input.channels}")
    print(f"True 2.5D: {config.input.use_true_2_5d}")
    print(f"\nTraining:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Optimizer: {config.training.optimizer.name}")
    print(f"  Learning rate: {config.training.optimizer.learning_rate}")
    print(f"\nSSM Config:")
    print(f"  State dim: {config.liquid_ssm.state_dim}")
    print(f"  Liquid modulation: {config.liquid_ssm.use_liquid_modulation}")
    print(f"\nKAN Config:")
    print(f"  Grid size: {config.kan.grid_size}")
    print(f"  TTT enabled: {config.kan.ttt_enabled}")
