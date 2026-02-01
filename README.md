# Brain Tumor Detection Using Deep Learning

**A Complete Deep Learning Solution for Detecting Brain Tumors from MRI Images**

An end-to-end deep learning solution for detecting brain tumors from MRI images, featuring a robust CNN architecture with data augmentation, comprehensive evaluation metrics, and production-ready code.

---

## 📖 Overview

This project implements a deep learning-based system for automated brain tumor detection from MRI scans. The system includes:

1. **Baseline CNN**: Custom 4-block convolutional neural network (~96% accuracy)
2. **Simple CNN**: Lightweight model for quick experiments
3. **Regularized CNN**: L2-regularized model for better generalization

### Key Highlights

- **Multiple Architecture Support**: Baseline, Simple, and Regularized CNNs
- **Comprehensive Preprocessing**: Data augmentation, CLAHE enhancement
- **Clinical Robustness**: Designed for real-world deployment
- **Complete Pipeline**: From data preprocessing to model deployment
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization Tools**: Training history, confusion matrices, ROC curves
- **Easy to Use**: One-click training and testing scripts

---

## ⚡ Quick Start (5 Minutes)

```bash
# 1. Clone and install
git clone https://github.com/Vikaash-dev/ai-brai-tumor.git
cd ai-brai-tumor
pip install -r requirements.txt

# 2. Prepare your dataset (see Dataset Setup section)
# Place MRI images in data/train/, data/validation/, data/test/

# 3. Train the model
python one_click_train_test.py --mode train --model-type baseline

# 4. Evaluate the model
python one_click_train_test.py --mode evaluate

# 5. Make predictions
python one_click_train_test.py --mode predict --image path/to/mri_image.jpg
```

---

## ✨ Features

### Core Features

- **Automated Brain Tumor Detection**: Binary classification (tumor vs. no tumor)
- **Multiple Architectures**: Baseline CNN, Simple CNN, Regularized CNN
- **Performance Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Visualization Tools**: Confusion matrices, ROC curves, training history plots
- **Model Checkpointing**: Saves best model during training
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Batch Prediction**: Process multiple images at once
- **Interactive Prediction**: Real-time prediction interface

### Training Features

- **Data Augmentation**: Rotation, shifts, shear, zoom, flip
- **Learning Rate Scheduling**: Automatic reduction on plateau
- **Class Weight Support**: Handle imbalanced datasets
- **TensorBoard Integration**: Optional training visualization

---

## 📁 Project Structure

```
ai-brai-tumor/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                           # Configuration parameters
├── one_click_train_test.py             # Main entry point
│
├── data/                               # Dataset directory
│   ├── train/
│   │   ├── tumor/                      # Tumor MRI images (training)
│   │   └── no_tumor/                   # Non-tumor MRI images (training)
│   ├── validation/
│   │   ├── tumor/                      # Tumor MRI images (validation)
│   │   └── no_tumor/                   # Non-tumor MRI images (validation)
│   └── test/
│       ├── tumor/                      # Tumor MRI images (testing)
│       └── no_tumor/                   # Non-tumor MRI images (testing)
│
├── models/                             # Model definitions
│   ├── __init__.py
│   ├── cnn_model.py                    # CNN architectures
│   └── saved_models/                   # Trained model files
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data loading and augmentation
│   ├── train.py                         # Training script
│   ├── evaluate.py                      # Evaluation and metrics
│   ├── predict.py                       # Prediction script
│   └── visualize.py                     # Visualization utilities
│
├── results/                            # Output directory
│   ├── confusion_matrix.png            # Confusion matrix plot
│   ├── roc_curve.png                   # ROC curve plot
│   ├── training_history.png            # Training history plots
│   ├── classification_report.txt       # Detailed metrics
│   └── batch_predictions.txt           # Batch prediction results
│
├── scripts/                            # Utility scripts
│
└── notebooks/                          # Jupyter notebooks
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
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

### Directory Structure

Organize your MRI images in the following structure:

```
data/
├── train/
│   ├── tumor/           # Tumor images for training
│   └── no_tumor/        # Non-tumor images for training
├── validation/
│   ├── tumor/           # Tumor images for validation
│   └── no_tumor/        # Non-tumor images for validation
└── test/
    ├── tumor/           # Tumor images for testing
    └── no_tumor/        # Non-tumor images for testing
```

### Recommended Datasets

You can use publicly available brain tumor datasets:

1. **Brain Tumor Classification (MRI)** - Kaggle
2. **Brain Tumor MRI Dataset** - Kaggle

### Data Split Recommendations

- **Training**: 70% of data
- **Validation**: 15% of data
- **Testing**: 15% of data

---

## 🔧 Usage

### One-Click Training and Testing

```bash
# Train with default settings
python one_click_train_test.py --mode train

# Train with custom settings
python one_click_train_test.py --mode train \
    --model-type regularized \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001

# Evaluate trained model
python one_click_train_test.py --mode evaluate

# Make single prediction
python one_click_train_test.py --mode predict --image path/to/image.jpg

# Full pipeline (train + evaluate)
python one_click_train_test.py --mode full --visualize
```

### Module-Level Usage

```bash
# Train model
python -m src.train --model-type baseline --epochs 50

# Evaluate model
python -m src.evaluate --model-path models/saved_models/best_model.h5

# Interactive prediction
python -m src.predict --interactive

# Batch prediction
python -m src.predict --directory ./test_images/ --output results.txt
```

### Python API Usage

```python
from models.cnn_model import get_model, load_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_single_image

# Train a model
model, history = train_model(model_type='baseline', epochs=50)

# Or load a pre-trained model
model = load_model('models/saved_models/best_model.h5')

# Evaluate
metrics = evaluate_model(model, test_dir='data/test')

# Make predictions
result = predict_single_image(model, 'path/to/mri_image.jpg')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🏗️ Model Architectures

### Baseline CNN

The baseline model uses a 4-block CNN architecture:

- **Block 1**: 32 filters, BatchNorm, MaxPool
- **Block 2**: 64 filters, BatchNorm, MaxPool
- **Block 3**: 128 filters, BatchNorm, MaxPool
- **Block 4**: 256 filters, BatchNorm, MaxPool
- **Head**: GlobalAvgPool → Dense(512) → Dense(256) → Softmax

### Simple CNN

Lightweight model for quick experiments:

- 3 convolutional blocks with increasing filters (32 → 64 → 128)
- Flatten → Dense(128) → Dropout → Softmax

### Regularized CNN

L2-regularized model for better generalization:

- Same structure as Baseline
- L2 regularization on all layers
- Additional dropout layers

---

## 📈 Results

### Expected Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline CNN | ~96% | ~95% | ~96% | ~95% |
| Simple CNN | ~92% | ~91% | ~92% | ~91% |
| Regularized CNN | ~95% | ~94% | ~95% | ~94% |

*Results may vary based on dataset and hyperparameters.*

---

## 📚 Configuration

Edit `config.py` to customize:

```python
# Image settings
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Training settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Augmentation settings
ROTATION_RANGE = 20
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
```

---

## 🔬 Research Papers

This implementation is based on research in:

- Deep Learning for Medical Image Analysis
- Convolutional Neural Networks for Brain Tumor Classification
- Data Augmentation Strategies for Medical Imaging

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- TensorFlow/Keras team for the excellent deep learning framework
- The medical imaging community for datasets and research
- Open source contributors