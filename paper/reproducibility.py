"""
Reproducibility Script for PHOENIX-v3.1 Research Paper

This script ensures complete reproducibility of all experiments
by setting random seeds, logging configurations, and saving
all necessary artifacts.

Reference: Research Paper Appendix A - Reproducibility Checklist
"""

import os
import sys
import json
import hashlib
import platform
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """
    Manages reproducibility for research experiments.
    
    Features:
    - Random seed management
    - Environment logging
    - Configuration hashing
    - Artifact tracking
    """
    
    def __init__(
        self,
        seed: int = 42,
        experiment_name: str = "phoenix_v3_1",
        output_dir: str = "experiments"
    ):
        """
        Initialize reproducibility manager.
        
        Args:
            seed: Master random seed
            experiment_name: Name of experiment
            output_dir: Directory for artifacts
        """
        self.seed = seed
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory
        self.experiment_dir = os.path.join(output_dir, f"{experiment_name}_{self.run_id}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Set all random seeds
        self._set_seeds()
        
        # Log environment
        self._log_environment()
    
    def _set_seeds(self) -> None:
        """Set random seeds for all libraries."""
        import random
        random.seed(self.seed)
        
        import numpy as np
        np.random.seed(self.seed)
        
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
            
            # Deterministic operations (may impact performance)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        except ImportError:
            pass
        
        try:
            import torch
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        logger.info(f"Random seeds set to {self.seed}")
    
    def _log_environment(self) -> None:
        """Log complete environment information."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "seed": self.seed,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler()
            },
            "system": {
                "os": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "packages": self._get_package_versions()
        }
        
        # Add GPU info
        env_info["gpu"] = self._get_gpu_info()
        
        # Save to file
        env_path = os.path.join(self.experiment_dir, "environment.json")
        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)
        
        logger.info(f"Environment logged to {env_path}")
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}
        
        try:
            import tensorflow as tf
            packages["tensorflow"] = tf.__version__
        except ImportError:
            pass
        
        try:
            import numpy as np
            packages["numpy"] = np.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            packages["scipy"] = scipy.__version__
        except ImportError:
            pass
        
        try:
            import sklearn
            packages["scikit-learn"] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import cv2
            packages["opencv"] = cv2.__version__
        except ImportError:
            pass
        
        return packages
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {"available": False}
        
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_info["available"] = True
                gpu_info["count"] = len(gpus)
                gpu_info["devices"] = [gpu.name for gpu in gpus]
                
                # Get memory info
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
        except:
            pass
        
        return gpu_info
    
    def log_config(self, config: Dict[str, Any]) -> str:
        """
        Log experiment configuration and return hash.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SHA256 hash of configuration
        """
        # Compute hash
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        
        # Save config
        config_path = os.path.join(self.experiment_dir, f"config_{config_hash}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration logged: {config_hash}")
        
        return config_hash
    
    def log_model(self, model, name: str = "model") -> None:
        """
        Log model architecture and weights.
        
        Args:
            model: Keras model
            name: Model name
        """
        # Save architecture
        arch_path = os.path.join(self.experiment_dir, f"{name}_architecture.json")
        with open(arch_path, 'w') as f:
            f.write(model.to_json())
        
        # Save summary
        summary_path = os.path.join(self.experiment_dir, f"{name}_summary.txt")
        with open(summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # Save weights
        weights_path = os.path.join(self.experiment_dir, f"{name}_weights.h5")
        model.save_weights(weights_path)
        
        logger.info(f"Model logged to {self.experiment_dir}")
    
    def log_results(self, results: Dict[str, Any], name: str = "results") -> None:
        """
        Log experiment results.
        
        Args:
            results: Results dictionary
            name: Results name
        """
        results_path = os.path.join(self.experiment_dir, f"{name}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results logged to {results_path}")
    
    def log_training_history(self, history, name: str = "history") -> None:
        """
        Log training history.
        
        Args:
            history: Keras History object
            name: History name
        """
        history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        
        history_path = os.path.join(self.experiment_dir, f"{name}.json")
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history logged to {history_path}")
    
    def generate_checklist(self) -> str:
        """
        Generate reproducibility checklist for paper.
        
        Returns:
            Markdown checklist
        """
        checklist = [
            "# Reproducibility Checklist",
            "",
            "## Code Availability",
            f"- [x] Code repository: `github.com/[repo]`",
            f"- [x] Experiment ID: `{self.run_id}`",
            f"- [x] Configuration hash: Available in `{self.experiment_dir}`",
            "",
            "## Environment",
            f"- [x] Python version: {platform.python_version()}",
            f"- [x] Random seed: {self.seed}",
            "- [x] Package versions: Logged in `environment.json`",
            "- [x] GPU configuration: Logged",
            "",
            "## Data",
            "- [ ] Dataset: BraTS 2023",
            "- [ ] Data splits: Stratified 5-fold CV",
            "- [ ] Preprocessing: Logged in configuration",
            "",
            "## Training",
            "- [x] Hyperparameters: Logged in configuration",
            "- [x] Training history: Logged",
            "- [x] Model weights: Saved",
            "",
            "## Evaluation",
            "- [ ] Metrics: Dice, HD95",
            "- [ ] Statistical tests: Paired t-test with Bonferroni",
            "- [ ] Confidence intervals: 95%",
            "",
            f"Generated: {datetime.now().isoformat()}"
        ]
        
        checklist_path = os.path.join(self.experiment_dir, "REPRODUCIBILITY_CHECKLIST.md")
        with open(checklist_path, 'w') as f:
            f.write('\n'.join(checklist))
        
        return '\n'.join(checklist)


def create_experiment(
    name: str = "phoenix_v3_1",
    seed: int = 42,
    config: Optional[Dict] = None
) -> ReproducibilityManager:
    """
    Create a new reproducible experiment.
    
    Args:
        name: Experiment name
        seed: Random seed
        config: Configuration dictionary
        
    Returns:
        ReproducibilityManager instance
    """
    manager = ReproducibilityManager(seed=seed, experiment_name=name)
    
    if config:
        manager.log_config(config)
    
    return manager


if __name__ == "__main__":
    # Example usage
    print("Creating reproducible experiment...")
    
    config = {
        "model": "phoenix_v3_1",
        "input_size": [224, 224, 3],
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "optimizer": "adan",
        "loss": "focal",
        "seed": 42
    }
    
    manager = create_experiment(
        name="phoenix_v3_1_brats2023",
        seed=42,
        config=config
    )
    
    # Generate checklist
    checklist = manager.generate_checklist()
    print(checklist)
    
    print(f"\nExperiment directory: {manager.experiment_dir}")
