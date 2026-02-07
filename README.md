# Deep Learning Radiomics (DLR) Pipeline for Medical Image Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Complete radiomics analysis pipeline for medical imaging, including feature extraction, LASSO feature selection, Deep Learning Radiomics (DLR) model training, and comprehensive evaluation.

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Overview](#-pipeline-overview)
- [Repository Structure](#-repository-structure)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [License](#-license)

## 🌟 Features

### Complete Analysis Pipeline

- ✅ **Image Preprocessing**
  - Resampling to uniform spacing
  - Intensity normalization
  - Label standardization

- ✅ **Feature Extraction**
  - 3D radiomics features (PyRadiomics)
  - Intratumoral and peritumoral features
  - Deep learning features (ResNet18, DINOv2)

- ✅ **LASSO Feature Selection**
  - Cross-validated optimal parameter selection
  - Feature importance ranking
  - Dimensionality reduction: 1000+ → 10-50 features

- ✅ **Model Training**
  - Logistic regression
  - Grid search hyperparameter tuning
  - 5-fold cross-validation

- ✅ **Model Evaluation**
  - ROC curves and AUC
  - Calibration curves
  - Confusion matrix
  - Comprehensive metrics

### Reproducibility Guarantees

- 🔒 Fixed random seeds
- 📊 Stratified cross-validation
- 💾 Saved models and scalers
- 📝 Complete documentation
- 🧪 Automated testing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Step-by-Step Execution

```bash
# Step 1: Feature extraction
python 3d_Radiomics_feature.py data/images data/masks
python concat_all_features.py

# Step 2: LASSO feature selection
python feature_selection_lasso.py features.csv labels.csv lasso_results/

# Step 3: Model training
python model_training_dlr.py lasso_results/features_lasso_selected.csv models/ 0.2

# Step 4: Model evaluation
python model_evaluation.py models/dlr_model.pkl models/dlr_scaler.pkl test_data.csv results/
```

## 📦 Installation

### System Requirements

- Python 3.8 or higher
- 8-16 GB RAM
- GPU (optional, for deep learning features)

### Dependencies

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0

# Medical imaging
SimpleITK>=2.1.0
pyradiomics>=3.0.1
nibabel>=3.2.0

# Deep learning
torch>=1.10.0
torchvision>=0.11.0
transformers>=4.20.0
```

See `requirements.txt` for complete list

## 📖 Usage

### Data Preparation

Prepare the following data:

```
project/
├── data/
│   ├── images/           # NIfTI images (.nii.gz)
│   ├── masks/            # NIfTI masks (.nii.gz)
│   └── labels.csv        # Label file
```

**Label file format:**

```csv
ID,label
patient001,0
patient002,1
patient003,1
```

### Configuration File

Create `config.json`:

```json
{
  "image_dir": "data/images",
  "mask_dir": "data/masks",
  "label_csv": "data/labels.csv",
  "feature_output_dir": "output/features",
  "lasso_output_dir": "output/lasso_results",
  "model_output_dir": "output/trained_models",
  "evaluation_output_dir": "output/evaluation",
  "extract_peritumoral": true,
  "extract_dl_features": true,
  "test_size": 0.2
}
```

### Examples

#### Example 1: Full Automated Pipeline

```bash
python pipeline_full.py config.json
```

#### Example 2: Feature Selection and Training Only

```bash
# Assuming features are already extracted
python feature_selection_lasso.py all_features.csv labels.csv output/
python model_training_dlr.py output/features_lasso_selected.csv models/ 0.2
```

#### Example 3: External Validation

```bash
python model_evaluation.py \
    models/dlr_model.pkl \
    models/dlr_scaler.pkl \
    external_test_data.csv \
    external_results/
```

## 🔄 Pipeline Overview

```
Input Data
    ↓
Image Preprocessing
    ↓
Feature Extraction
  ├─ 3D Radiomics
  ├─ Peritumoral Features
  └─ Deep Learning Features
    ↓
Feature Concatenation
    ↓
LASSO Feature Selection
    ↓
Model Training
  └─ Hyperparameter Tuning
    ↓
Model Evaluation
  ├─ ROC Curve
  ├─ Calibration Curve
  └─ Performance Metrics
    ↓
Final Output
```

See [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) for detailed diagram

## 📁 Repository Structure

```
├── Image Preprocessing
│   ├── step1_image_preprocess.py
│   ├── step3_mask_preprocess.py
│   ├── regis.py
│   └── regis2024.py
│
├── Feature Extraction
│   ├── 3d_Radiomics_feature.py
│   ├── Intratumoral_peritumoral_feature.py
│   ├── dinov2_feature.py
│   ├── max_roi_resnet18_radio_feature.py
│   └── concat_all_features.py
│
├── Feature Selection
│   └── feature_selection_lasso.py
│
├── Model Training
│   └── model_training_dlr.py
│
├── Model Evaluation
│   └── model_evaluation.py
│
├── Complete Pipeline
│   └── pipeline_full.py
│
├── Documentation
│   ├── README.md (this file)
│   ├── README_MODEL_TRAINING.md
│   ├── QUICK_START.md
│   ├── WORKFLOW_DIAGRAM.md
│   └── requirements.txt
│
└── Testing
    ├── test_pipeline.py
    └── config_example.json
```

## 📚 Documentation

- **[README_MODEL_TRAINING.md](README_MODEL_TRAINING.md)** - Complete technical documentation
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)** - Visual workflow
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Implementation summary

## 🔬 Method Details

### LASSO Feature Selection

- **Algorithm**: LassoCV with cross-validation
- **Cross-validation**: 5-fold stratified CV
- **Alpha range**: 100 automatically determined values
- **Standardization**: StandardScaler (zero mean, unit variance)

### DLR Model Training

- **Algorithm**: Logistic Regression
- **Hyperparameter tuning**: Grid search with CV
- **Tuned parameters**:
  - Regularization strength C: [0.001, 0.01, 0.1, 1, 10, 100]
  - Penalty: ['l1', 'l2']
  - Solver: ['liblinear', 'saga']
- **Scoring metric**: ROC AUC
- **Class weights**: Balanced

### Evaluation Metrics

- ROC AUC
- Accuracy
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)
- Average Precision
- Calibration Curve

## 📊 Output Results

### LASSO Selection Results

- `selected_features_lasso.csv` - Selected features with coefficients
- `features_lasso_selected.csv` - Filtered dataset
- `lasso_model.pkl` - LASSO model
- `scaler.pkl` - Feature scaler

### Model Training Results

- `dlr_model.pkl` - Trained model
- `dlr_scaler.pkl` - Feature scaler
- `cv_results.csv` - Cross-validation results
- `test_predictions.csv` - Test predictions
- `model_summary.csv` - Performance summary

### Evaluation Results

- `roc_curve.png` - ROC curve plot
- `pr_curve.png` - Precision-Recall curve
- `calibration_curve.png` - Calibration plot
- `confusion_matrix.png` - Confusion matrix
- `evaluation_metrics.csv` - All metrics
- `classification_report.csv` - Classification report

## 🧪 Testing

Run automated tests:

```bash
python test_pipeline.py
```

Tests create synthetic data and verify all pipeline components.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or suggestions, please contact:

- Open an issue: [GitHub Issues](https://github.com/CHANGHAI-AILab/AP)
- Email: timchen91@aliyun.com


## 🙏 Acknowledgments

- PyRadiomics team for excellent radiomics tools
- scikit-learn team for machine learning library
- All open-source contributors

## 📈 Changelog

### v1.0.0 (2024)

- ✨ Initial release
- ✅ Complete feature extraction pipeline
- ✅ LASSO feature selection
- ✅ DLR model training and evaluation
- ✅ Complete documentation and testing

---

**⭐ If this project helps you, please give us a star!**




