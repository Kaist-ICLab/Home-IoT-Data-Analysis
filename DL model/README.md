# 🏠 Multimodal Mental Health Detection using Deep Learning

This repository contains a deep learning model for multimodal mental health detection, specifically focusing on PHQ-2 and GAD-2 assessments using various data modalities including IoT, voice, phone, and wearable sensor data.

## 📁 Project Structure

```
├── DATASET/            # Merged dataset from FEATURES directory
├── FEATURES/          # Individual feature sets from different modalities
├── Funcs/            # Utility functions
├── RESULTS/          # Model results and outputs
├── main.py           # Main training and evaluation script
├── models.py         # Model architectures
├── dataset_setup.py  # Dataset preparation utilities
├── customdataset.py  # Custom dataset implementation
├── visualize.py      # Visualization utilities
└── requirements.txt  # Project dependencies
```

## 🏗️ Model Architectures

The project implements several deep learning architectures:
- `CNN1d`: 1D Convolutional Neural Network
- `FusionBase`: Multi-modal fusion model combining different data modalities

## ⚙️ Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Additional dependencies (not listed in requirements.txt):
- PyTorch
- scikit-learn
- pandas
- numpy

## 📊 Dataset

The dataset is a merged version of features from the `FEATURES/` directory, which contains:
- IoT sensor data
- Voice features
- Phone usage patterns
- Wearable sensor data

The merged dataset is stored in the `DATASET/` directory and is automatically processed during training.

## 🚀 Usage

### Running Complete Experiments

To run the complete experiments for both PHQ-2 and GAD-2 assessments:

```bash
# Run PHQ-2 experiments
bash run_phq_all.sh

# Run GAD-2 experiments
bash run_gad_all.sh
```

### Single Model Training

To train a single model:

```bash
# PHQ-2 binary classification
python main.py --label phq2_result_binary --splitter loso --modelname fusion

# GAD-2 binary classification
python main.py --label gad2_result_binary --splitter loso --modelname conv1d
```

### Command Line Arguments

- `--label`: Target label (e.g., phq2_result_binary, gad2_result_binary)
- `--splitter`: Data splitting strategy (loso: Leave-One-Subject-Out, personalstratifiedkfold)
- `--modelname`: Model architecture (conv1d, conv1dattn, fusion)
- `--mixup`: Enable/disable mixup data augmentation
- `--seed`: Random seed for reproducibility

## ✨ Features

- Multi-modal data fusion
- Leave-One-Subject-Out (LOSO) cross-validation
- Personal stratified k-fold cross-validation
- Mixup data augmentation
- SHAP value analysis for model interpretability
- Comprehensive evaluation metrics (Accuracy, AUC, F1, Balanced Accuracy)

## 📈 Results

Model results and visualizations are saved in the `RESULTS/` directory.
