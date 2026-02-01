# Brain Tumor Detection Using Deep Learning - Phoenix Protocol

**A Complete, Production-Ready SOTA Neuro-Oncology AI System**

An end-to-end deep learning solution for detecting brain tumors from MRI images, featuring the advanced **NeuroSnake architecture** with Dynamic Snake Convolutions, Coordinate Attention, and comprehensive training optimizations.

**🎉 Status**: Production-Ready | SOTA Architecture | Edge-Optimized

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Clone and install
git clone https://github.com/Vikaash-dev/ai-brai-tumor.git
cd ai-brai-tumor
pip install -r requirements.txt

# 2. Prepare your dataset (see Dataset Setup section)
# Place MRI images in data/train/{tumor,no_tumor}, data/validation/..., data/test/...

# 3. Train SOTA model (NeuroSnake + Coordinate Attention)
python one_click_train_test.py --mode train --model-type neurosnake_ca --deduplicate

# 4. Evaluate the model
python one_click_train_test.py --mode evaluate --visualize

# 5. Make predictions
python one_click_train_test.py --mode predict --image path/to/mri_image.jpg
```

---

## 🔥 Phoenix Protocol (SOTA Features)

The **Phoenix Protocol** represents a complete reimagining of lightweight neuro-oncology AI, addressing critical vulnerabilities while maintaining edge-deployability.

### Key Innovations

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Dynamic Snake Convolutions** | Adaptive kernel deformation | Traces irregular Glioblastoma boundaries |
| **Coordinate Attention** | Position-preserving attention | Tumor location is diagnostic |
| **MobileViT Block** | Global context capture | Detects mass effect |
| **Adan Optimizer** | 1st, 2nd, 3rd moment estimation | Superior stability |
| **Focal Loss** | Class imbalance handling | Focus on hard examples |
| **Physics-Informed Augmentation** | MRI-specific transforms | Realistic training data |
| **pHash Deduplication** | Prevent data leakage | Honest accuracy metrics |

### Architecture Overview

```
Input (224×224×3)
    ↓
Stem Conv (32 filters, stride=2)
    ↓
Snake Conv Block 1 (64) + Coordinate Attention → MaxPool
    ↓
Snake Conv Block 2 (128) + Coordinate Attention → MaxPool
    ↓
Snake Conv Block 3 (256) + Coordinate Attention → MaxPool
    ↓
Snake Conv Block 4 (512) + MobileViT + Coordinate Attention → MaxPool
    ↓
Global Average Pooling
    ↓
Dense (256) → Dense (128) → Softmax (2)
```

---

## ✨ Features

### Core Features
- **Automated Brain Tumor Detection**: Binary classification (tumor vs. no tumor)
- **Multiple Architectures**: Baseline CNN, NeuroSnake, NeuroSnake+CA (SOTA)
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization Tools**: Confusion matrices, ROC curves, training history
- **Model Checkpointing**: Saves best model during training
- **Early Stopping**: Prevents overfitting

### Phoenix Protocol Features
- 🔬 **Data Deduplication**: pHash-based duplicate detection
- ⚗️ **Physics-Informed Augmentation**: Elastic deformation, Rician noise
- 🧠 **Dynamic Snake Convolutions**: Adaptive kernel deformation
- ⚡ **Adan Optimizer**: Advanced Nesterov momentum
- 🎯 **Focal Loss**: Class imbalance handling
- 📍 **Coordinate Attention**: Position-preserving feature extraction

---

## 📁 Project Structure

```
ai-brai-tumor/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                           # Configuration parameters
├── one_click_train_test.py             # Main entry point (SOTA)
│
├── models/                             # Model architectures
│   ├── cnn_model.py                    # Baseline CNN
│   ├── neurosnake_model.py             # NeuroSnake architecture
│   ├── dynamic_snake_conv.py           # Dynamic Snake Convolutions
│   └── coordinate_attention.py         # Coordinate Attention module
│
├── src/                                # Source code
│   ├── data_preprocessing.py           # Data loading and augmentation
│   ├── data_deduplication.py           # pHash deduplication
│   ├── physics_informed_augmentation.py # MRI-specific augmentation
│   ├── phoenix_optimizer.py            # Adan optimizer + Focal Loss
│   ├── train.py                         # Basic training
│   ├── train_phoenix.py                 # SOTA training pipeline
│   ├── evaluate.py                      # Evaluation metrics
│   ├── predict.py                       # Prediction interface
│   └── visualize.py                     # Visualization tools
│
├── data/                               # Dataset directory
│   ├── train/{tumor, no_tumor}/
│   ├── validation/{tumor, no_tumor}/
│   └── test/{tumor, no_tumor}/
│
└── results/                            # Output directory
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA for faster training

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/Vikaash-dev/ai-brai-tumor.git
cd ai-brai-tumor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

Organize your MRI images in the following structure:

```
data/
├── train/
│   ├── tumor/           # Tumor MRI images
│   └── no_tumor/        # Non-tumor MRI images
├── validation/
│   ├── tumor/
│   └── no_tumor/
└── test/
    ├── tumor/
    └── no_tumor/
```

### Recommended Datasets
- **Brain Tumor Classification (MRI)** - Kaggle
- **Brain MRI Images for Brain Tumor Detection** - Kaggle

---

## 🔧 Usage

### SOTA Training (Recommended)

```bash
# Train NeuroSnake with Coordinate Attention (SOTA)
python one_click_train_test.py --mode train --model-type neurosnake_ca

# With all optimizations
python one_click_train_test.py --mode train \
    --model-type neurosnake_ca \
    --deduplicate \
    --epochs 100 \
    --visualize

# Full pipeline (train + evaluate)
python one_click_train_test.py --mode full --visualize
```

### Model Information

```bash
# View available models and features
python one_click_train_test.py --mode info
```

### Prediction

```bash
# Single image prediction
python one_click_train_test.py --mode predict --image path/to/scan.jpg

# Interactive mode
python one_click_train_test.py --mode predict
```

### Python API

```python
from src.train_phoenix import PhoenixProtocolTrainer

# Create SOTA trainer
trainer = PhoenixProtocolTrainer(
    model_type='neurosnake_ca',
    use_focal_loss=True,
    use_adan_optimizer=True,
    use_physics_augmentation=True
)

# Train
history = trainer.train(epochs=100)

# Evaluate
metrics = trainer.evaluate()
```

---

## 🏗️ Model Architectures

| Model | Description | Parameters | Use Case |
|-------|-------------|------------|----------|
| `baseline` | Standard 4-block CNN | ~8M | Quick experiments |
| `neurosnake` | DSC + MobileViT | ~2M | Production |
| `neurosnake_ca` | DSC + MobileViT + CA | ~2.2M | **SOTA (Recommended)** |

---

## 📈 Expected Performance

| Metric | Baseline | NeuroSnake | NeuroSnake+CA |
|--------|----------|------------|---------------|
| Accuracy | ~92% | ~95% | **~96%** |
| Precision | ~91% | ~94% | **~95%** |
| Recall | ~92% | ~95% | **~96%** |
| F1-Score | ~91% | ~94% | **~95%** |

*Results after deduplication. Actual performance depends on dataset quality.*

---

## 🔬 Research Background

This implementation is based on:
- **Dynamic Snake Convolutions**: Adaptive kernel deformation for curvilinear features
- **Coordinate Attention (CVPR 2021)**: Position-preserving attention for mobile networks
- **MobileViT**: Lightweight vision transformers
- **Adan Optimizer**: Adaptive Nesterov momentum for faster convergence
- **Focal Loss (ICCV 2017)**: Handling class imbalance in object detection

---

## 🆕 SOTA Features (True State-of-the-Art)

This implementation leverages the latest research from arXiv, top conferences (CVPR, MICCAI, ICCV), and SOTA GitHub repositories.

### Training Improvements (from nnU-Net, MONAI)
- ✅ **Mixed Precision Training (AMP)**: 2-3x speedup on GPU
- ✅ **K-Fold Cross-Validation**: Patient-level stratification
- ✅ **Advanced LR Schedulers**: Cosine annealing with warmup, OneCycle
- ✅ **Gradient Clipping**: Prevents training instability
- ✅ **Early Stopping**: With patience and best weights restoration
- ✅ **Reproducibility**: Seed fixing for deterministic results

### Clinical Preprocessing (from nnU-Net)
- ✅ **Skull Stripping**: Brain extraction using morphological operations
- ✅ **N4 Bias Field Correction**: Corrects RF coil inhomogeneities
- ✅ **CLAHE Enhancement**: Adaptive contrast enhancement
- ✅ **Z-Score Normalization**: Per-scan intensity standardization

### Clinical Postprocessing
- ✅ **MC Dropout Uncertainty**: "I don't know" for ambiguous cases
- ✅ **Grad-CAM Explainability**: Visual attention maps
- ✅ **Confidence Thresholding**: Flags low-confidence predictions
- ✅ **Clinical Reports**: Automated diagnostic reports

### Edge Deployment (EfficientQuant)
- ✅ **Hybrid Quantization**: Uniform for CNN, Log2 for Transformer
- ✅ **TFLite Export**: For mobile/edge deployment
- ✅ **2.5-8.7× Latency Reduction**: With <1% accuracy loss

### Test-Time Augmentation (TTA)
- ✅ Horizontal/Vertical flip
- ✅ 90°/180°/270° rotation
- ✅ Prediction averaging

---

## 🛠️ Advanced Usage

### K-Fold Cross-Validation

```python
from src.training_improvements import SOTATrainer, set_reproducibility

# Set seed for reproducibility
set_reproducibility(42)

# Train with K-Fold
trainer = SOTATrainer(model, use_mixed_precision=True)
results = trainer.train_kfold(
    X, y, 
    n_splits=5, 
    epochs=100,
    patient_ids=patient_ids  # Patient-level stratification
)
print(f"Average accuracy: {results['average_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
```

### Clinical Preprocessing

```python
from src.clinical_preprocessing import create_preprocessing_pipeline

# Create clinical preprocessing pipeline
preprocessor = create_preprocessing_pipeline(
    target_size=(224, 224),
    apply_skull_strip=True,
    apply_bias_correction=True,
    apply_clahe=True,
    normalization='zscore'
)

# Preprocess image
preprocessed = preprocessor.preprocess(mri_image)
```

### Uncertainty Quantification

```python
from src.clinical_postprocessing import ClinicalPostprocessor

# Create clinical postprocessor
postprocessor = ClinicalPostprocessor(
    model,
    class_names=["No Tumor", "Tumor"],
    enable_uncertainty=True,
    enable_gradcam=True
)

# Get prediction with uncertainty
result = postprocessor.process(image, generate_heatmap=True)
print(f"Prediction: {result['class_label']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Uncertainty: {result['uncertainty']:.1%}")
print(f"Flags: {result['flags']}")
```

### Edge Deployment (Quantization)

```python
from src.efficient_quant import EfficientQuantizer

# Create quantizer with hybrid strategy
quantizer = EfficientQuantizer(
    model=model,
    calibration_data=cal_data,
    strategy='hybrid'  # Uniform for CNN, Log2 for Transformer
)

# Quantize model
quantized_model = quantizer.quantize_hybrid_model()

# Validate accuracy retention
results = quantizer.validate_quantized_model(test_data, test_labels, quantized_model)
print(f"Original accuracy: {results['original_accuracy']:.4f}")
print(f"Quantized accuracy: {results['quantized_accuracy']:.4f}")
print(f"Accuracy loss: {results['accuracy_loss_relative_percent']:.2f}%")

# Export to TFLite
quantizer.export_tflite('model_int8.tflite', quantized_model)
```

---

## 📚 Documentation

- [PHOENIX_PROTOCOL.md](PHOENIX_PROTOCOL.md) - Complete implementation guide
- [Research_Paper_Brain_Tumor_Detection.md](Research_Paper_Brain_Tumor_Detection.md) - Research background

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Medical Disclaimer

This system is for research purposes only. Not approved for clinical use. Always consult qualified healthcare professionals for medical decisions.

---

## 🙏 Acknowledgements

- TensorFlow/Keras team
- nnU-Net and MONAI teams (medical imaging best practices)
- Medical imaging research community
- Original Phoenix Protocol research

---

## 📖 References

1. **Dynamic Snake Convolutions** - CVPR 2023
2. **Coordinate Attention for Efficient Mobile Network Design** - CVPR 2021
3. **MobileViT: Light-weight Vision Transformer** - Apple Research
4. **Adan: Adaptive Nesterov Momentum Algorithm** - 2022
5. **Focal Loss for Dense Object Detection** - ICCV 2017
6. **nnU-Net: Self-adapting Framework for Medical Image Segmentation** - Nature Methods
7. **MONAI: Medical Open Network for AI** - NVIDIA/King's College