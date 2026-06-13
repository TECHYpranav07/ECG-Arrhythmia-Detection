# ECG-Based Arrhythmia Detection using Capsule Networks (CapsNet) & CNN Models

This repository contains an advanced Deep Learning project for detecting Cardiac Arrhythmia from raw 12-lead ECG paper sheet images. The project implements and compares transfer learning baseline *[...]

It features a complete pipeline that crops raw paper ECG sheets into individual leads, processes them, runs predictions, and outputs a diagnostic recommendation using a majority voting scheme.

---

## 📌 Project Overview & Rationale

Electrocardiograms (ECGs) are primary clinical tools for diagnosing cardiac anomalies. While traditional CNNs excel at image classification, they lack spatial invariance—pooling layers discard c[...]

**Capsule Networks (CapsNet)** address this limitation by replacing scalar neurons with vector capsules that encode both the presence of a feature and its spatial properties (like angle, size, and[...]

---

## 📺 Demo Video

Here is a screen recording demonstrating the preprocessing, lead segmentation, and model prediction workflow:

<video src="Screen Recording 2026-03-24 211253.mp4" width="100%" controls></video>

---

## 📊 Dataset Structure & Statistics

The dataset consists of raw ECG sheets with a standard resolution of `2213x1572` pixels, categorized into two main groups:
1. **Normal Heartbeats**: 284 patient ECG sheets.
2. **Abnormal Heartbeats**: 233 patient ECG sheets.

### Lead Segmentation Pipeline
Each raw ECG sheet is preprocessed and cropped into **13 individual lead images** (12 standard leads + 1 long Lead II rhythm strip at the bottom):
- **12 Standard Leads**: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
- **1 Rhythmic Lead**: Lead II Rhythm Strip (used for heart rate variability and rhythm analysis)

This segmentation multiplies the dataset size to train the classification models:
- **Normal Leads**: $284 \text{ sheets} \times 13 \text{ leads} = 3,692$ images.
- **Abnormal Leads**: $233 \text{ sheets} \times 13 \text{ leads} = 3,029$ images.
- **Total Dataset**: **6,721 individual lead images**.

---

## 🧠 Model Architectures

### 1. Capsule Network (CapsNet)
The custom Capsule Network is designed with the following layers:
* **Convolutional Layer**: Conv2D (64 filters, 9x9 kernel, stride 1, valid padding) + Batch Normalization + ReLU.
* **Primary Capsule Layer**: Maps features into 8-dimensional vector capsules. Compiles $h \times w \times 8$ capsules, each of dimension 8.
* **Digit Capsule Layer**: Maps input capsules to 2 class capsules (representing Abnormal and Normal classes), each of dimension 16.
* **Dynamic Routing**: Uses 3 routing iterations to determine coupling coefficients between primary and digit capsules.
* **Squash Activation**: Compresses vector lengths to values between 0 and 1, representing class probability.

### 2. CNN Baselines (Transfer Learning)
The CNN baseline notebook evaluates and compares several pre-trained state-of-the-art CNN architectures:
* **VGG16** (Validation Accuracy: **98.86%**)
* **ResNet50** (Validation Accuracy: **98.86%**)
* **Xception**
* **InceptionV3**
* **EfficientNetB0**

*Note: Transfer learning backbones are frozen, and classification is performed via Global Average Pooling followed by custom Dense layers.*

---

## 📈 Training Results & Performance

* **CapsNet Performance**:
  - Training Accuracy: **~99.14%**
  - Validation Accuracy: **~97.32%**
  - Precision: **0.99** (Abnormal), **0.95** (Normal)
  - Recall: **0.94** (Abnormal), **0.99** (Normal)
* **CNN Baseline Performance**:
  - The CNN models also achieve high accuracies (~98%), but require larger image inputs (`224x224x3`) and have orders of magnitude more parameters than CapsNet, which processes `128x128x1` graysca[...]

---

## 📁 Repository Structure

```
├── ECG Images of Patient that have abnormal heartbeat (233x12=2796)/ # Raw Abnormal ECG Sheets
├── Normal Person ECG Images (284x12=3408)/                           # Raw Normal ECG Sheets
├── Research-paper/                                                    # Academic papers & reference PDFs
├── src/                                                               # Production scripts
│   ├── preprocess.py                                                 # Lead segmentation CLI
│   └── predict.py                                                    # End-to-end inference CLI
├── CAPSNET_work_with_preprocess_predict.ipynb                         # Capsule Network experiments
├── CAPSNET_work_with_preprocess_predict_dynamic_routng.ipynb          # CapsNet + Routing
├── CAPSNET_work_with_preprocess_predict_dynamic_routng_99_percent.ipynb# Best-performing CapsNet model
├── CNNs_(1)_UPDATED (1).ipynb                                         # CNN Baselines
├── requirements.txt                                                   # Project requirements
├── Screen Recording 2026-03-24 211253.mp4                             # Demo video
└── README.md                                                          # This documentation
```

---

## 🚀 Getting Started & Usage

### 1. Installation
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Run Training Notebooks
You can train either model by opening the respective Jupyter Notebook in your environment (Local Jupyter or Google Colab):
- `CAPSNET_work_with_preprocess_predict_dynamic_routng_99_percent.ipynb` for the best Capsule Network.
- `CNNs_(1)_UPDATED (1).ipynb` for the Transfer Learning CNN benchmarks.

*Note: The notebooks are preconfigured to save training splits (`X_train.npy`, etc.) to skip preprocessing on subsequent runs.*

### 3. Extract Leads from a Raw ECG Sheet
Use `preprocess.py` to segment a patient's full ECG sheet into 13 individual lead images. If the sheet's resolution is not `2213x1572`, the script will automatically resize it for exact coordinat[...]
```bash
python src/preprocess.py --image "path/to/raw_patient_ecg.jpg" --outdir "temp_leads"
```

### 4. Perform End-to-End Classification (weighted majority voting)
Pass the folder of extracted leads and a trained model file to `predict.py`. The script will predict normal/abnormal for each of the 13 leads, run a majority vote, and print the diagnosis conclus[...]
```bash
python src/predict.py --model "path/to/trained_capsnet.h5" --leaddir "temp_leads"
```

---

## 📝 Diagnostic Majority Voting Scheme
Because an ECG represents heart activity from multiple spatial angles, a single lead containing an abnormality indicates general arrhythmia. The pipeline performs **weighted majority voting**:
- If the majority ($\ge 7$) of the 13 leads are normal, the patient status is declared **NORMAL**.
- Otherwise, the patient is flagged as **ABNORMAL ARRHYTHMIA DETECTED**, prompting clinical review.
