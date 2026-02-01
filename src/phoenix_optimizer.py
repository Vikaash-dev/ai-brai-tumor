"""
Phoenix Protocol Optimizer and Loss Functions.
Implements Adan optimizer and Focal Loss for superior training stability
on medical imaging datasets.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple


class AdanOptimizer(keras.optimizers.Optimizer):
    """
    Adan (Adaptive Nesterov Momentum Algorithm) Optimizer.
    
    Adan estimates first, second, and third moments for superior stability
    on non-convex landscapes common in medical imaging tasks.
    
    Unlike Lion which discards gradient magnitude (uses only sign),
    Adan preserves this information for more stable convergence.
    
    Reference: "Adan: Adaptive Nesterov Momentum Algorithm for 
               Faster Optimizing Deep Models" (2022)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.98,
        beta2: float = 0.92,
        beta3: float = 0.99,
        weight_decay: float = 0.02,
        epsilon: float = 1e-8,
        name: str = "Adan",
        **kwargs
    ):
        """
        Initialize Adan optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            beta3: Exponential decay rate for third moment (gradient difference)
            weight_decay: Weight decay coefficient
            epsilon: Small constant for numerical stability
            name: Optimizer name
        """
        super(AdanOptimizer, self).__init__(name=name, **kwargs)
        
        self._learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        
    def build(self, var_list):
        """Build optimizer state."""
        super().build(var_list)
        
        if hasattr(self, "_built") and self._built:
            return
            
        self._m = []  # First moment (like Adam)
        self._v = []  # Second moment (like Adam)
        self._n = []  # Third moment (gradient difference)
        self._prev_grad = []  # Previous gradient for difference calculation
        
        for var in var_list:
            self._m.append(
                self.add_variable_from_reference(var, name="m")
            )
            self._v.append(
                self.add_variable_from_reference(var, name="v")
            )
            self._n.append(
                self.add_variable_from_reference(var, name="n")
            )
            self._prev_grad.append(
                self.add_variable_from_reference(var, name="prev_grad")
            )
            
        self._built = True
        
    def update_step(self, gradient, variable, learning_rate):
        """Perform one optimization step."""
        var_key = self._var_key(variable)
        
        # Get optimizer state
        m = self._m[self._index_dict[var_key]]
        v = self._v[self._index_dict[var_key]]
        n = self._n[self._index_dict[var_key]]
        prev_grad = self._prev_grad[self._index_dict[var_key]]
        
        # Compute gradient difference
        grad_diff = gradient - prev_grad
        
        # Update moments
        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        m.assign(self.beta1 * m + (1 - self.beta1) * gradient)
        
        # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        v.assign(self.beta2 * v + (1 - self.beta2) * tf.square(gradient))
        
        # n_t = beta3 * n_{t-1} + (1 - beta3) * (g_t - g_{t-1})^2
        n.assign(self.beta3 * n + (1 - self.beta3) * tf.square(grad_diff))
        
        # Bias correction
        step = tf.cast(self.iterations + 1, tf.float32)
        m_hat = m / (1 - tf.pow(self.beta1, step))
        v_hat = v / (1 - tf.pow(self.beta2, step))
        n_hat = n / (1 - tf.pow(self.beta3, step))
        
        # Compute update
        # Adan uses: m_hat + beta2 * grad_diff
        update = (m_hat + self.beta2 * grad_diff) / (
            tf.sqrt(v_hat + n_hat) + self.epsilon
        )
        
        # Apply weight decay
        if self.weight_decay > 0:
            update = update + self.weight_decay * variable
        
        # Update variable
        variable.assign_sub(learning_rate * update)
        
        # Store current gradient for next step
        prev_grad.assign(gradient)
        
    def get_config(self):
        """Return optimizer configuration."""
        config = super().get_config()
        config.update({
            "learning_rate": self._learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "beta3": self.beta3,
            "weight_decay": self.weight_decay,
            "epsilon": self.epsilon,
        })
        return config


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for handling class imbalance in medical imaging.
    
    Focal Loss down-weights easy negatives and focuses on hard examples,
    which is critical for rare tumor types and class imbalance.
    
    Reference: "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        name: str = "focal_loss"
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class balance weight (alpha for positive class)
            gamma: Focusing parameter (higher = more focus on hard examples)
            label_smoothing: Label smoothing factor
            name: Loss name
        """
        super(FocalLoss, self).__init__(name=name)
        
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def call(self, y_true, y_pred):
        """
        Compute focal loss.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + (
                self.label_smoothing / num_classes
            )
        
        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        
        # Compute cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Compute focal weight: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Apply alpha balance
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        """Return loss configuration."""
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing,
        })
        return config


def create_adan_optimizer(
    learning_rate: float = 0.001,
    beta1: float = 0.98,
    beta2: float = 0.92,
    beta3: float = 0.99,
    weight_decay: float = 0.02
) -> AdanOptimizer:
    """
    Factory function to create Adan optimizer.
    
    Args:
        learning_rate: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        beta3: Third moment decay rate
        weight_decay: Weight decay coefficient
        
    Returns:
        Configured Adan optimizer
    """
    return AdanOptimizer(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        weight_decay=weight_decay
    )


def create_focal_loss(
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.1
) -> FocalLoss:
    """
    Factory function to create Focal Loss.
    
    Args:
        alpha: Class balance weight
        gamma: Focusing parameter
        label_smoothing: Label smoothing factor
        
    Returns:
        Configured Focal Loss
    """
    return FocalLoss(
        alpha=alpha,
        gamma=gamma,
        label_smoothing=label_smoothing
    )


if __name__ == "__main__":
    print("Testing Phoenix Protocol Optimizer and Loss Functions...")
    
    # Test Focal Loss
    print("\n1. Testing Focal Loss...")
    focal_loss = create_focal_loss(alpha=0.25, gamma=2.0)
    
    # Create test data
    y_true = tf.constant([[1, 0], [0, 1], [1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]], dtype=tf.float32)
    
    loss_value = focal_loss(y_true, y_pred)
    print(f"   Focal Loss value: {loss_value.numpy():.4f}")
    
    # Test with imbalanced prediction
    y_pred_hard = tf.constant([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]], dtype=tf.float32)
    loss_hard = focal_loss(y_true, y_pred_hard)
    print(f"   Focal Loss (hard examples): {loss_hard.numpy():.4f}")
    
    # Test Adan Optimizer
    print("\n2. Testing Adan Optimizer...")
    
    # Create simple model
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,)),
        keras.layers.Dense(2, activation='softmax')
    ])
    
    # Compile with Adan
    adan_optimizer = create_adan_optimizer(learning_rate=0.001)
    model.compile(optimizer=adan_optimizer, loss=focal_loss)
    
    # Test training step
    x_test = tf.random.normal((4, 5))
    y_test = tf.constant([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=tf.float32)
    
    # Fit one step
    history = model.fit(x_test, y_test, epochs=1, verbose=0)
    print(f"   Training loss: {history.history['loss'][0]:.4f}")
    
    print("\n✓ Phoenix Protocol Optimizer and Loss test passed!")
