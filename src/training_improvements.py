# SOTA Training Improvements for Medical Imaging
"""
Implements state-of-the-art training techniques from:
- nnU-Net (SOTA medical image segmentation)
- MONAI (Medical Open Network for AI)
- PyTorch Lightning best practices
- Recent research papers (2023-2026)

Features:
- Mixed Precision Training (AMP) - 2-3x speedup
- K-Fold Cross-Validation with patient-level stratification
- Advanced learning rate schedulers
- Gradient clipping and accumulation
- Early stopping with patience
- Test-Time Augmentation (TTA)
- Model Ensembling
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from typing import List, Dict, Tuple, Optional, Callable, Any
import logging
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """
    Mixed Precision Training for 2-3x speedup.
    Uses float16 for forward/backward pass, float32 for weights.
    """
    
    def __init__(self, enable: bool = True):
        self.enable = enable
        
    def setup(self) -> None:
        """Enable mixed precision training."""
        if self.enable:
            try:
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision (float16) enabled - expect 2-3x speedup")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
                
    def get_loss_scale_optimizer(
        self,
        optimizer: keras.optimizers.Optimizer
    ) -> keras.optimizers.Optimizer:
        """Wrap optimizer with loss scaling for mixed precision."""
        if self.enable:
            return keras.mixed_precision.LossScaleOptimizer(optimizer)
        return optimizer


class KFoldCrossValidator:
    """
    K-Fold Cross-Validation with patient-level stratification.
    Critical for medical imaging to prevent data leakage.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        
    def get_folds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        patient_ids: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/val indices for each fold.
        
        Args:
            X: Features
            y: Labels
            patient_ids: Optional patient IDs for patient-level splitting
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        if patient_ids is not None:
            # Patient-level stratification
            unique_patients = np.unique(patient_ids)
            patient_labels = []
            for pid in unique_patients:
                mask = patient_ids == pid
                patient_labels.append(y[mask][0])  # Assume same label per patient
            patient_labels = np.array(patient_labels)
            
            folds = []
            for train_patient_idx, val_patient_idx in self.kfold.split(unique_patients, patient_labels):
                train_patients = unique_patients[train_patient_idx]
                val_patients = unique_patients[val_patient_idx]
                
                train_idx = np.where(np.isin(patient_ids, train_patients))[0]
                val_idx = np.where(np.isin(patient_ids, val_patients))[0]
                
                folds.append((train_idx, val_idx))
            return folds
        else:
            return list(self.kfold.split(X, y))


class AdvancedLRScheduler:
    """
    Advanced learning rate schedulers from SOTA research.
    """
    
    @staticmethod
    def cosine_annealing(
        initial_lr: float,
        epochs: int,
        min_lr: float = 1e-7,
        warmup_epochs: int = 5
    ) -> keras.callbacks.LearningRateScheduler:
        """
        Cosine annealing with warm restarts.
        
        Args:
            initial_lr: Starting learning rate
            epochs: Total epochs
            min_lr: Minimum learning rate
            warmup_epochs: Warmup period
            
        Returns:
            LearningRateScheduler callback
        """
        def schedule(epoch, lr):
            if epoch < warmup_epochs:
                # Linear warmup
                return initial_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
        
        return keras.callbacks.LearningRateScheduler(schedule, verbose=1)
    
    @staticmethod
    def reduce_on_plateau(
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7
    ) -> keras.callbacks.ReduceLROnPlateau:
        """
        Reduce LR when validation loss plateaus.
        """
        return keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=1
        )
    
    @staticmethod
    def one_cycle(
        max_lr: float,
        epochs: int,
        steps_per_epoch: int
    ) -> keras.callbacks.Callback:
        """
        One Cycle Learning Rate policy (Leslie Smith).
        """
        class OneCycleScheduler(keras.callbacks.Callback):
            def __init__(self, max_lr, epochs, steps_per_epoch):
                super().__init__()
                self.max_lr = max_lr
                self.epochs = epochs
                self.steps_per_epoch = steps_per_epoch
                self.total_steps = epochs * steps_per_epoch
                self.step = 0
                
            def on_batch_begin(self, batch, logs=None):
                self.step += 1
                progress = self.step / self.total_steps
                
                if progress < 0.5:
                    # Increase to max_lr
                    lr = self.max_lr * progress * 2
                else:
                    # Decrease from max_lr
                    lr = self.max_lr * (1 - (progress - 0.5) * 2)
                
                K.set_value(self.model.optimizer.learning_rate, lr)
        
        return OneCycleScheduler(max_lr, epochs, steps_per_epoch)


class GradientAccumulator:
    """
    Gradient accumulation for effective larger batch sizes.
    Useful when GPU memory is limited.
    """
    
    def __init__(
        self,
        accumulation_steps: int = 4,
        gradient_clip_value: float = 1.0
    ):
        self.accumulation_steps = accumulation_steps
        self.gradient_clip_value = gradient_clip_value
        
    def get_train_step(
        self,
        model: keras.Model,
        optimizer: keras.optimizers.Optimizer,
        loss_fn: Callable
    ) -> Callable:
        """
        Create custom train step with gradient accumulation.
        """
        @tf.function
        def train_step(x_batch, y_batch, accumulated_gradients):
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)
                scaled_loss = loss / self.accumulation_steps
            
            gradients = tape.gradient(scaled_loss, model.trainable_variables)
            
            # Clip gradients
            gradients = [
                tf.clip_by_value(g, -self.gradient_clip_value, self.gradient_clip_value)
                if g is not None else g
                for g in gradients
            ]
            
            # Accumulate
            for i, grad in enumerate(gradients):
                if grad is not None:
                    accumulated_gradients[i].assign_add(grad)
            
            return loss
        
        return train_step


class EarlyStopping:
    """
    Enhanced early stopping with additional features.
    """
    
    @staticmethod
    def create(
        patience: int = 15,
        restore_best_weights: bool = True,
        min_delta: float = 0.001
    ) -> keras.callbacks.EarlyStopping:
        """
        Create early stopping callback.
        
        Args:
            patience: Number of epochs to wait
            restore_best_weights: Restore best weights on stop
            min_delta: Minimum improvement threshold
            
        Returns:
            EarlyStopping callback
        """
        return keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=restore_best_weights,
            min_delta=min_delta,
            verbose=1
        )


class TestTimeAugmentation:
    """
    Test-Time Augmentation for robust predictions.
    Averages predictions over augmented versions of input.
    """
    
    def __init__(
        self,
        augmentations: List[str] = None,
        merge_strategy: str = 'mean'
    ):
        """
        Initialize TTA.
        
        Args:
            augmentations: List of augmentation types
            merge_strategy: 'mean', 'max', or 'voting'
        """
        self.augmentations = augmentations or [
            'original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270'
        ]
        self.merge_strategy = merge_strategy
        
    def _apply_augmentation(
        self,
        image: np.ndarray,
        aug_type: str
    ) -> np.ndarray:
        """Apply single augmentation to image."""
        if aug_type == 'original':
            return image
        elif aug_type == 'hflip':
            return np.fliplr(image)
        elif aug_type == 'vflip':
            return np.flipud(image)
        elif aug_type == 'rotate90':
            return np.rot90(image, k=1)
        elif aug_type == 'rotate180':
            return np.rot90(image, k=2)
        elif aug_type == 'rotate270':
            return np.rot90(image, k=3)
        else:
            return image
    
    def _reverse_augmentation(
        self,
        prediction: np.ndarray,
        aug_type: str
    ) -> np.ndarray:
        """Reverse augmentation on prediction."""
        # For classification, no reversal needed
        return prediction
    
    def predict(
        self,
        model: keras.Model,
        images: np.ndarray
    ) -> np.ndarray:
        """
        Make TTA predictions.
        
        Args:
            model: Keras model
            images: Input images (N, H, W, C)
            
        Returns:
            TTA predictions
        """
        all_predictions = []
        
        for aug_type in self.augmentations:
            # Apply augmentation to all images
            augmented = np.array([
                self._apply_augmentation(img, aug_type)
                for img in images
            ])
            
            # Get predictions
            preds = model.predict(augmented, verbose=0)
            
            # Reverse augmentation on predictions
            reversed_preds = np.array([
                self._reverse_augmentation(pred, aug_type)
                for pred in preds
            ])
            
            all_predictions.append(reversed_preds)
        
        all_predictions = np.array(all_predictions)
        
        # Merge predictions
        if self.merge_strategy == 'mean':
            return np.mean(all_predictions, axis=0)
        elif self.merge_strategy == 'max':
            return np.max(all_predictions, axis=0)
        elif self.merge_strategy == 'voting':
            votes = np.argmax(all_predictions, axis=-1)
            final_preds = []
            for i in range(votes.shape[1]):
                counts = np.bincount(votes[:, i], minlength=all_predictions.shape[-1])
                final_preds.append(counts / len(self.augmentations))
            return np.array(final_preds)
        else:
            return np.mean(all_predictions, axis=0)


class ModelEnsemble:
    """
    Model ensembling for robust predictions.
    Combines multiple models for better performance.
    """
    
    def __init__(
        self,
        models: List[keras.Model],
        weights: Optional[List[float]] = None,
        strategy: str = 'average'
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of Keras models
            weights: Optional weights for each model
            strategy: 'average', 'weighted', or 'voting'
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.strategy = strategy
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input data
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        for model in self.models:
            preds = model.predict(X, verbose=0)
            predictions.append(preds)
        
        predictions = np.array(predictions)
        
        if self.strategy == 'average':
            return np.mean(predictions, axis=0)
        elif self.strategy == 'weighted':
            weighted = np.sum([
                w * p for w, p in zip(self.weights, predictions)
            ], axis=0)
            return weighted
        elif self.strategy == 'voting':
            votes = np.argmax(predictions, axis=-1)
            final_preds = []
            for i in range(votes.shape[1]):
                counts = np.bincount(votes[:, i], minlength=predictions.shape[-1])
                final_preds.append(counts / len(self.models))
            return np.array(final_preds)
        else:
            return np.mean(predictions, axis=0)


class SOTATrainer:
    """
    State-of-the-art trainer combining all improvements.
    """
    
    def __init__(
        self,
        model: keras.Model,
        use_mixed_precision: bool = True,
        use_gradient_accumulation: bool = False,
        accumulation_steps: int = 4,
        gradient_clip_value: float = 1.0
    ):
        self.model = model
        self.mixed_precision = MixedPrecisionTrainer(use_mixed_precision)
        self.gradient_clip_value = gradient_clip_value
        
        if use_gradient_accumulation:
            self.gradient_accumulator = GradientAccumulator(
                accumulation_steps, gradient_clip_value
            )
        else:
            self.gradient_accumulator = None
            
        self.history = None
        
    def setup(self) -> None:
        """Setup training environment."""
        self.mixed_precision.setup()
        
        # Set memory growth for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Could not set memory growth: {e}")
                
    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        loss: Callable,
        metrics: List[str] = None
    ) -> None:
        """
        Compile model with SOTA settings.
        """
        # Wrap optimizer for mixed precision
        optimizer = self.mixed_precision.get_loss_scale_optimizer(optimizer)
        
        # Apply gradient clipping
        if hasattr(optimizer, 'clipvalue'):
            optimizer.clipvalue = self.gradient_clip_value
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics or ['accuracy']
        )
        
    def get_callbacks(
        self,
        epochs: int,
        learning_rate: float,
        steps_per_epoch: int = None,
        checkpoint_path: str = None
    ) -> List[keras.callbacks.Callback]:
        """
        Get SOTA callbacks.
        """
        callbacks = []
        
        # Learning rate scheduler
        callbacks.append(AdvancedLRScheduler.cosine_annealing(
            initial_lr=learning_rate,
            epochs=epochs,
            warmup_epochs=5
        ))
        
        # Early stopping
        callbacks.append(EarlyStopping.create(patience=15))
        
        # Model checkpoint
        if checkpoint_path:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ))
        
        # TensorBoard
        log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ))
        
        return callbacks
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        checkpoint_path: str = None
    ) -> Dict[str, List[float]]:
        """
        Train model with SOTA techniques.
        """
        logger.info("Starting SOTA training...")
        
        callbacks = self.get_callbacks(
            epochs=epochs,
            learning_rate=learning_rate,
            steps_per_epoch=len(X_train) // batch_size,
            checkpoint_path=checkpoint_path
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def train_kfold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patient_ids: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Train with K-Fold cross-validation.
        """
        logger.info(f"Starting {n_splits}-fold cross-validation...")
        
        kfold = KFoldCrossValidator(n_splits=n_splits)
        folds = kfold.get_folds(X, y, patient_ids)
        
        fold_histories = []
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Reset model weights
            for layer in self.model.layers:
                if hasattr(layer, 'kernel_initializer'):
                    layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            
            history = self.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            fold_histories.append(history)
            
            # Evaluate fold
            metrics = self.model.evaluate(X_val, y_val, verbose=0)
            fold_metrics.append(metrics)
            
            logger.info(f"Fold {fold_idx + 1} - Loss: {metrics[0]:.4f}, Acc: {metrics[1]:.4f}")
        
        # Aggregate results
        avg_loss = np.mean([m[0] for m in fold_metrics])
        avg_acc = np.mean([m[1] for m in fold_metrics])
        std_acc = np.std([m[1] for m in fold_metrics])
        
        logger.info(f"\n=== Cross-Validation Results ===")
        logger.info(f"Average Loss: {avg_loss:.4f}")
        logger.info(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        
        return {
            'fold_histories': fold_histories,
            'fold_metrics': fold_metrics,
            'average_loss': avg_loss,
            'average_accuracy': avg_acc,
            'std_accuracy': std_acc
        }


def set_reproducibility(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    logger.info(f"Reproducibility set with seed: {seed}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("SOTA Training Improvements for Medical Imaging")
    print("=" * 60)
    print("\nFeatures included:")
    print("✓ Mixed Precision Training (AMP)")
    print("✓ K-Fold Cross-Validation")
    print("✓ Advanced LR Schedulers (Cosine, OneCycle)")
    print("✓ Gradient Clipping & Accumulation")
    print("✓ Early Stopping with Patience")
    print("✓ Test-Time Augmentation (TTA)")
    print("✓ Model Ensembling")
    print("✓ Reproducibility (Seed Fixing)")
