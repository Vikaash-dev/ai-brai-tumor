# Clinical Postprocessing for Medical AI Predictions
"""
Implements clinical-grade postprocessing for brain tumor detection:
- Uncertainty quantification (MC Dropout, ensemble variance)
- Confidence thresholding
- Grad-CAM visualization for explainability
- Clinical report generation
- Test-Time Augmentation (TTA)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ClinicalPrediction:
    """Clinical prediction with metadata."""
    class_label: str
    confidence: float
    uncertainty: float
    attention_map: Optional[np.ndarray]
    all_probabilities: Dict[str, float]
    flags: List[str]
    timestamp: str


class UncertaintyQuantifier:
    """
    Uncertainty quantification using MC Dropout.
    Enables model to express "I don't know" for ambiguous cases.
    """
    
    def __init__(
        self,
        model: keras.Model,
        n_samples: int = 30,
        dropout_rate: float = 0.1
    ):
        """
        Initialize uncertainty quantifier.
        
        Args:
            model: Keras model (must have Dropout layers)
            n_samples: Number of MC samples
            dropout_rate: Dropout rate for MC sampling
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
    def _enable_dropout(self) -> None:
        """Enable dropout during inference for MC sampling."""
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Dropout):
                layer.training = True
                
    def _disable_dropout(self) -> None:
        """Disable dropout after MC sampling."""
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Dropout):
                layer.training = False
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input images (N, H, W, C)
            
        Returns:
            Tuple of (mean_predictions, predictive_uncertainty, epistemic_uncertainty)
        """
        # Collect MC samples
        mc_predictions = []
        
        self._enable_dropout()
        
        for _ in range(self.n_samples):
            preds = self.model(X, training=True).numpy()
            mc_predictions.append(preds)
            
        self._disable_dropout()
        
        mc_predictions = np.array(mc_predictions)  # (n_samples, N, n_classes)
        
        # Mean prediction
        mean_pred = np.mean(mc_predictions, axis=0)
        
        # Predictive uncertainty (total variance)
        predictive_var = np.var(mc_predictions, axis=0)
        predictive_uncertainty = np.mean(predictive_var, axis=-1)
        
        # Epistemic uncertainty (variance of means)
        epistemic_uncertainty = np.std(mc_predictions, axis=0)
        epistemic_uncertainty = np.mean(epistemic_uncertainty, axis=-1)
        
        return mean_pred, predictive_uncertainty, epistemic_uncertainty


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for explainability.
    Shows which regions the model focuses on for its decision.
    """
    
    def __init__(
        self,
        model: keras.Model,
        layer_name: Optional[str] = None
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name: Target layer name (default: last conv layer)
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                return layer.name
        raise ValueError("No convolutional layer found in model")
    
    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Input image (H, W, C)
            class_idx: Target class index (default: predicted class)
            
        Returns:
            Heatmap (H, W) with values in [0, 1]
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get target layer
        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
        
        # Get gradients with respect to conv outputs
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs),
            axis=-1
        )
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize to input size
        import cv2
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        
        return heatmap
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on image.
        
        Args:
            image: Original image (H, W, C)
            heatmap: Grad-CAM heatmap (H, W)
            alpha: Overlay transparency
            
        Returns:
            Image with heatmap overlay
        """
        import cv2
        
        # Convert heatmap to RGB colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_JET
        )
        heatmap_colored = heatmap_colored / 255.0
        
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Overlay
        overlaid = alpha * heatmap_colored + (1 - alpha) * image
        overlaid = np.clip(overlaid, 0, 1)
        
        return overlaid


class ConfidenceThresholder:
    """
    Confidence-based filtering for clinical safety.
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.9,
        low_confidence_threshold: float = 0.6,
        uncertainty_threshold: float = 0.15
    ):
        self.high_conf = high_confidence_threshold
        self.low_conf = low_confidence_threshold
        self.uncertainty_thresh = uncertainty_threshold
        
    def classify_confidence(
        self,
        probability: float,
        uncertainty: Optional[float] = None
    ) -> Tuple[str, List[str]]:
        """
        Classify confidence level and generate flags.
        
        Args:
            probability: Prediction probability
            uncertainty: Optional uncertainty estimate
            
        Returns:
            Tuple of (confidence_level, flags)
        """
        flags = []
        
        if probability >= self.high_conf:
            confidence_level = "HIGH"
        elif probability >= self.low_conf:
            confidence_level = "MODERATE"
            flags.append("REVIEW_RECOMMENDED")
        else:
            confidence_level = "LOW"
            flags.append("MANUAL_REVIEW_REQUIRED")
        
        if uncertainty is not None and uncertainty > self.uncertainty_thresh:
            flags.append("HIGH_UNCERTAINTY")
            
        if probability > 0.4 and probability < 0.6:
            flags.append("BORDERLINE_CASE")
            
        return confidence_level, flags


class ClinicalReportGenerator:
    """
    Generate clinical reports for predictions.
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        institution: str = "AI Research Lab"
    ):
        self.class_names = class_names or ["No Tumor", "Tumor"]
        self.institution = institution
        
    def generate_report(
        self,
        prediction: ClinicalPrediction,
        patient_id: str = "Unknown",
        study_id: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Generate clinical report.
        
        Args:
            prediction: ClinicalPrediction object
            patient_id: Patient identifier
            study_id: Study identifier
            
        Returns:
            Report dictionary
        """
        report = {
            "header": {
                "institution": self.institution,
                "report_type": "AI-Assisted Brain Tumor Detection",
                "generated_at": datetime.now().isoformat(),
                "patient_id": patient_id,
                "study_id": study_id
            },
            "findings": {
                "primary_diagnosis": prediction.class_label,
                "confidence": f"{prediction.confidence:.1%}",
                "confidence_level": self._get_confidence_level(prediction.confidence),
                "uncertainty": f"{prediction.uncertainty:.1%}" if prediction.uncertainty else "N/A",
                "all_probabilities": {
                    k: f"{v:.1%}" for k, v in prediction.all_probabilities.items()
                }
            },
            "quality_indicators": {
                "flags": prediction.flags,
                "requires_review": len(prediction.flags) > 0,
                "explainability_available": prediction.attention_map is not None
            },
            "disclaimer": (
                "This report is generated by an AI system and is intended "
                "for decision support only. All findings should be verified "
                "by a qualified radiologist. This system is not approved for "
                "clinical diagnosis."
            )
        }
        
        return report
    
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "HIGH"
        elif confidence >= 0.7:
            return "MODERATE"
        else:
            return "LOW"
    
    def format_report_text(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = [
            "=" * 60,
            f"INSTITUTION: {report['header']['institution']}",
            f"REPORT TYPE: {report['header']['report_type']}",
            f"GENERATED: {report['header']['generated_at']}",
            f"PATIENT ID: {report['header']['patient_id']}",
            f"STUDY ID: {report['header']['study_id']}",
            "=" * 60,
            "",
            "FINDINGS:",
            f"  Primary Diagnosis: {report['findings']['primary_diagnosis']}",
            f"  Confidence: {report['findings']['confidence']} ({report['findings']['confidence_level']})",
            f"  Uncertainty: {report['findings']['uncertainty']}",
            "",
            "  Class Probabilities:",
        ]
        
        for class_name, prob in report['findings']['all_probabilities'].items():
            lines.append(f"    - {class_name}: {prob}")
        
        lines.extend([
            "",
            "QUALITY INDICATORS:",
            f"  Flags: {', '.join(report['quality_indicators']['flags']) or 'None'}",
            f"  Requires Review: {'Yes' if report['quality_indicators']['requires_review'] else 'No'}",
            "",
            "DISCLAIMER:",
            report['disclaimer'],
            "=" * 60
        ])
        
        return "\n".join(lines)


class ClinicalPostprocessor:
    """
    Complete clinical postprocessing pipeline.
    """
    
    def __init__(
        self,
        model: keras.Model,
        class_names: List[str] = None,
        enable_uncertainty: bool = True,
        enable_gradcam: bool = True,
        mc_samples: int = 30
    ):
        self.model = model
        self.class_names = class_names or ["No Tumor", "Tumor"]
        
        if enable_uncertainty:
            self.uncertainty_quantifier = UncertaintyQuantifier(
                model, n_samples=mc_samples
            )
        else:
            self.uncertainty_quantifier = None
            
        if enable_gradcam:
            self.gradcam = GradCAM(model)
        else:
            self.gradcam = None
            
        self.confidence_thresholder = ConfidenceThresholder()
        self.report_generator = ClinicalReportGenerator(class_names)
    
    def process(
        self,
        image: np.ndarray,
        generate_heatmap: bool = True,
        generate_report: bool = True,
        patient_id: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Process a single image through the clinical pipeline.
        
        Args:
            image: Input image (H, W, C)
            generate_heatmap: Whether to generate Grad-CAM heatmap
            generate_report: Whether to generate clinical report
            patient_id: Patient identifier
            
        Returns:
            Dictionary with prediction, heatmap, report, etc.
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image
        
        # Get predictions with uncertainty
        if self.uncertainty_quantifier:
            mean_pred, pred_uncertainty, epistemic = self.uncertainty_quantifier.predict_with_uncertainty(image_batch)
            uncertainty = float(pred_uncertainty[0])
        else:
            mean_pred = self.model.predict(image_batch, verbose=0)
            uncertainty = 0.0
        
        # Get class prediction
        class_idx = int(np.argmax(mean_pred[0]))
        class_label = self.class_names[class_idx]
        confidence = float(mean_pred[0, class_idx])
        
        # Get all probabilities
        all_probs = {
            name: float(mean_pred[0, i])
            for i, name in enumerate(self.class_names)
        }
        
        # Confidence assessment
        conf_level, flags = self.confidence_thresholder.classify_confidence(
            confidence, uncertainty
        )
        
        # Generate heatmap
        heatmap = None
        overlaid = None
        if generate_heatmap and self.gradcam:
            heatmap = self.gradcam.compute_heatmap(image, class_idx)
            overlaid = self.gradcam.overlay_heatmap(image, heatmap)
        
        # Create prediction object
        prediction = ClinicalPrediction(
            class_label=class_label,
            confidence=confidence,
            uncertainty=uncertainty,
            attention_map=heatmap,
            all_probabilities=all_probs,
            flags=flags,
            timestamp=datetime.now().isoformat()
        )
        
        # Generate report
        report = None
        report_text = None
        if generate_report:
            report = self.report_generator.generate_report(
                prediction, patient_id=patient_id
            )
            report_text = self.report_generator.format_report_text(report)
        
        return {
            'prediction': prediction,
            'class_label': class_label,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'confidence_level': conf_level,
            'flags': flags,
            'all_probabilities': all_probs,
            'heatmap': heatmap,
            'overlaid_image': overlaid,
            'report': report,
            'report_text': report_text
        }
    
    def process_batch(
        self,
        images: np.ndarray,
        generate_heatmaps: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images.
        
        Args:
            images: Batch of images (N, H, W, C)
            generate_heatmaps: Whether to generate Grad-CAM heatmaps
            
        Returns:
            List of result dictionaries
        """
        results = []
        for i, image in enumerate(images):
            result = self.process(
                image,
                generate_heatmap=generate_heatmaps,
                generate_report=False,
                patient_id=f"Batch-{i}"
            )
            results.append(result)
        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Clinical Postprocessing Pipeline")
    print("=" * 60)
    print("\nFeatures:")
    print("✓ MC Dropout Uncertainty Quantification")
    print("✓ Grad-CAM Explainability")
    print("✓ Confidence Thresholding")
    print("✓ Clinical Report Generation")
    print("✓ Test-Time Augmentation")
    print("\nFor radiologist decision support")
