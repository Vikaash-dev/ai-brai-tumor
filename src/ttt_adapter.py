"""
TTT (Test-Time Training) Adapter for PHOENIX-v3.1

Implements Symbolic Mirror TTT which adapts KAN spline weights
at inference time using entropy minimization.

Reference: Section 3.5 - Symbolic Mirror TTT Adapter
"""

import tensorflow as tf
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SymbolicMirrorTTT:
    """
    Test-Time Training adapter that updates KAN spline weights.
    
    The Symbolic Mirror concept:
    - At inference, create a self-supervised objective from prediction entropy
    - Only update spline weights (not backbone)
    - This "bends" the symbolic activation functions to match patient-specific
      intensity distributions
    
    θ_spline* = argmin_θ H(P(y|x; θ))
    
    Example:
        >>> ttt = SymbolicMirrorTTT(model)
        >>> adapted_pred = ttt.adapt_and_predict(patient_scan)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: float = 0.001,
        max_steps: int = 5,
        entropy_threshold: float = 0.3,
        early_stop_delta: float = 0.001,
        spline_layer_names: Optional[List[str]] = None
    ):
        """
        Initialize TTT adapter.
        
        Args:
            model: Keras model with KAN layers
            learning_rate: Learning rate for spline updates
            max_steps: Maximum adaptation steps
            entropy_threshold: Only adapt if entropy > threshold
            early_stop_delta: Stop if entropy improvement < delta
            spline_layer_names: Names of layers containing splines (auto-detect if None)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.entropy_threshold = entropy_threshold
        self.early_stop_delta = early_stop_delta
        
        # Find spline weights
        self.spline_weights = self._find_spline_weights(spline_layer_names)
        
        # Create optimizer for splines only
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Store original weights for reset
        self._original_weights = None
        
        logger.info(f"TTT Adapter initialized with {len(self.spline_weights)} spline weight tensors")
    
    def _find_spline_weights(
        self,
        layer_names: Optional[List[str]] = None
    ) -> List[tf.Variable]:
        """
        Find spline-related weights in model.
        
        Args:
            layer_names: Specific layer names to search
            
        Returns:
            List of spline weight variables
        """
        spline_weights = []
        
        for layer in self.model.layers:
            # Check if this is a KAN layer
            layer_name = layer.name.lower()
            
            # Match KAN-related layers
            is_kan_layer = (
                'kan' in layer_name or
                'spline' in layer_name or
                (layer_names and any(n in layer_name for n in layer_names))
            )
            
            if is_kan_layer:
                for weight in layer.trainable_weights:
                    weight_name = weight.name.lower()
                    # Only include spline-related weights
                    if 'spline' in weight_name or 'coef' in weight_name:
                        spline_weights.append(weight)
        
        # If no spline weights found, fall back to all layer weights
        # that match naming pattern
        if not spline_weights:
            for weight in self.model.trainable_weights:
                if 'spline' in weight.name.lower():
                    spline_weights.append(weight)
        
        return spline_weights
    
    def save_weights(self) -> None:
        """Save current spline weights for later reset."""
        self._original_weights = [w.numpy().copy() for w in self.spline_weights]
    
    def restore_weights(self) -> None:
        """Restore spline weights to saved state."""
        if self._original_weights is not None:
            for weight, original in zip(self.spline_weights, self._original_weights):
                weight.assign(original)
    
    @tf.function
    def compute_entropy(self, predictions: tf.Tensor) -> tf.Tensor:
        """
        Compute Shannon entropy of predictions.
        
        H(p) = -Σ p_i * log(p_i)
        
        Args:
            predictions: Softmax predictions [B, C]
            
        Returns:
            Mean entropy across batch
        """
        # Clip to avoid log(0)
        predictions = tf.clip_by_value(predictions, 1e-7, 1.0 - 1e-7)
        
        # Shannon entropy
        entropy = -tf.reduce_sum(predictions * tf.math.log(predictions), axis=-1)
        
        return tf.reduce_mean(entropy)
    
    @tf.function
    def adaptation_step(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Single adaptation step.
        
        Args:
            x: Input tensor
            
        Returns:
            (predictions, entropy)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=False)
            
            # Compute entropy loss
            entropy = self.compute_entropy(predictions)
        
        # Compute gradients only for spline weights
        if self.spline_weights:
            gradients = tape.gradient(entropy, self.spline_weights)
            
            # Apply gradients
            valid_grads = [
                (g, w) for g, w in zip(gradients, self.spline_weights)
                if g is not None
            ]
            
            if valid_grads:
                self.optimizer.apply_gradients(valid_grads)
        
        return predictions, entropy
    
    def adapt_and_predict(
        self,
        x: tf.Tensor,
        return_entropy_history: bool = False
    ) -> tf.Tensor:
        """
        Adapt model to input and return prediction.
        
        Args:
            x: Input tensor [B, H, W, C]
            return_entropy_history: If True, return entropy values
            
        Returns:
            Adapted predictions (and optionally entropy history)
        """
        # Save original weights
        self.save_weights()
        
        # Initial prediction
        initial_pred = self.model(x, training=False)
        initial_entropy = self.compute_entropy(initial_pred).numpy()
        
        # Check if adaptation is needed
        if initial_entropy < self.entropy_threshold:
            logger.debug(f"Entropy {initial_entropy:.4f} below threshold, skipping TTT")
            if return_entropy_history:
                return initial_pred, [initial_entropy]
            return initial_pred
        
        # Adaptation loop
        entropy_history = [initial_entropy]
        best_entropy = initial_entropy
        
        for step in range(self.max_steps):
            predictions, entropy = self.adaptation_step(x)
            entropy_val = entropy.numpy()
            entropy_history.append(entropy_val)
            
            # Check for early stopping
            improvement = best_entropy - entropy_val
            if improvement < self.early_stop_delta:
                logger.debug(f"TTT converged at step {step+1}")
                break
            
            best_entropy = min(best_entropy, entropy_val)
        
        # Final prediction
        final_pred = self.model(x, training=False)
        
        # Restore original weights for next sample
        self.restore_weights()
        
        if return_entropy_history:
            return final_pred, entropy_history
        return final_pred
    
    def adapt_batch(
        self,
        batch: tf.Tensor,
        per_sample: bool = True
    ) -> tf.Tensor:
        """
        Adapt and predict for a batch.
        
        Args:
            batch: Batch of inputs [B, H, W, C]
            per_sample: If True, adapt separately for each sample
            
        Returns:
            Predictions for batch
        """
        if per_sample:
            predictions = []
            for i in range(batch.shape[0]):
                sample = batch[i:i+1]
                pred = self.adapt_and_predict(sample)
                predictions.append(pred)
            return tf.concat(predictions, axis=0)
        else:
            return self.adapt_and_predict(batch)


class LazyTTT(SymbolicMirrorTTT):
    """
    Lazy TTT variant that only adapts when confidence is low.
    
    This is more efficient as it skips adaptation for easy cases.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        confidence_threshold: float = 0.8,
        **kwargs
    ):
        """
        Initialize Lazy TTT.
        
        Args:
            model: Keras model
            confidence_threshold: Only adapt if max confidence < threshold
            **kwargs: Additional arguments for SymbolicMirrorTTT
        """
        super().__init__(model, **kwargs)
        self.confidence_threshold = confidence_threshold
    
    def adapt_and_predict(
        self,
        x: tf.Tensor,
        return_entropy_history: bool = False
    ) -> tf.Tensor:
        """
        Lazily adapt model to input.
        
        Args:
            x: Input tensor
            return_entropy_history: If True, return entropy values
            
        Returns:
            Predictions
        """
        # Initial prediction
        initial_pred = self.model(x, training=False)
        max_confidence = tf.reduce_max(initial_pred).numpy()
        
        # Check if adaptation is needed
        if max_confidence >= self.confidence_threshold:
            logger.debug(f"Confidence {max_confidence:.4f} >= threshold, skipping TTT")
            if return_entropy_history:
                return initial_pred, []
            return initial_pred
        
        # Use parent's adaptation
        return super().adapt_and_predict(x, return_entropy_history)


class TTTWrapper(tf.keras.Model):
    """
    Keras Model wrapper that applies TTT during inference.
    
    This allows TTT to be used transparently with model.predict().
    """
    
    def __init__(
        self,
        base_model: tf.keras.Model,
        ttt_config: dict = None
    ):
        """
        Initialize TTT wrapper.
        
        Args:
            base_model: Base Keras model
            ttt_config: TTT configuration dictionary
        """
        super().__init__()
        self.base_model = base_model
        
        ttt_config = ttt_config or {}
        self.ttt = SymbolicMirrorTTT(
            base_model,
            learning_rate=ttt_config.get('learning_rate', 0.001),
            max_steps=ttt_config.get('max_steps', 5),
            entropy_threshold=ttt_config.get('entropy_threshold', 0.3)
        )
        
        self._use_ttt = True
    
    def call(self, inputs, training=None):
        """Forward pass with optional TTT."""
        if training:
            return self.base_model(inputs, training=True)
        
        if self._use_ttt:
            return self.ttt.adapt_and_predict(inputs)
        else:
            return self.base_model(inputs, training=False)
    
    def enable_ttt(self):
        """Enable TTT during inference."""
        self._use_ttt = True
    
    def disable_ttt(self):
        """Disable TTT during inference."""
        self._use_ttt = False
    
    @property
    def trainable_weights(self):
        return self.base_model.trainable_weights
    
    @property
    def non_trainable_weights(self):
        return self.base_model.non_trainable_weights


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Create a simple model with "spline" weights for testing
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Simulate KAN layer with spline weights
    spline_layer = tf.keras.layers.Dense(64, name='kan_spline_layer')
    x = spline_layer(x)
    
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Create TTT adapter
    ttt = SymbolicMirrorTTT(
        model,
        learning_rate=0.001,
        max_steps=3
    )
    
    # Test adaptation
    test_input = tf.random.normal([1, 224, 224, 3])
    
    print("Testing TTT adaptation...")
    pred, entropy_history = ttt.adapt_and_predict(test_input, return_entropy_history=True)
    
    print(f"Prediction shape: {pred.shape}")
    print(f"Entropy history: {entropy_history}")
    print("✅ TTT adapter working!")
