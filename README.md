# 🏠 Home IoT-based Depression & Anxiety Detection

This repository supports the paper: > ** ... **

We present a machine learning and deep learning framework that uses multimodal data collected from **mobile**, **wearable deivices**, and **home IoT sensors** to detect signs of **depression and anxiety** in real-world environments.

---

## 🚀 Getting Started

We provide extracted and preprocessed features (in ML models/FEATURES or DL models/FEATURES) to support reproducibility.

🔒 The raw data (from mobile, wearable, and IoT sensors) is not publicly released due to privacy concerns.


## 🤖 Model Training

### Traditional ML Models
* Location: ML_models/
* Models: Decision Tree, Random Forest, AdaBoost, XGBoost, LDA, kNN, SVM

```bash
cd ML_models/
python main.py
```
More details in ML_models/README.md

### Deep Learning Models
Location: DL_models/
Models: 1D-CNN, attention-based fusion

```bash
cd DL_models/
python main.py
```
More details in DL_models/README.md

## 📊 Evaluation
Includes:
* Accuracy, AUROC
* Generalized vs Personalized model comparison
* Top behavioral features analysis

##  📌 Notes
* Extracted features (after preprocessing) are shared.
* Raw sensor data is not included due to participant privacy constraints.
* The full pipeline can still be run end-to-end using the provided features.

## 📄 Citation
To be added upon publication.