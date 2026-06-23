# 🛡️ Adaptive Scam Detection in Data Streams Using ARF + ADWIN + C-SMOTE

> **PSM 1 Research Project — Universiti Teknologi Malaysia (UTM)**
> Faculty of Computing · Semester 2, 2025/2026

[![Python](https://img.shields.io/badge/Python-3.14.3-blue?logo=python&logoColor=white)](https://www.python.org/)
[![River](https://img.shields.io/badge/River-0.24.2-teal?logo=river&logoColor=white)](https://riverml.xyz/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-PSM%201%20Complete-brightgreen)]()

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Research Objectives](#-research-objectives)
- [System Architecture](#-system-architecture)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Methodology](#-methodology)
- [Evaluation Metrics](#-evaluation-metrics)
- [PSM 1 Results](#-psm-1-results)
- [PSM 2 Roadmap](#-psm-2-roadmap)
- [Tech Stack](#-tech-stack)
- [References](#-references)
- [Author](#-author)

---

## 🔍 About the Project

This research develops an **adaptive machine learning framework** for detecting digital scams in real-time data streams. Unlike traditional static models that are trained once and never updated, the proposed system continuously adapts to **evolving fraud patterns** without interrupting live predictions.

The system targets three distinct scam domains evaluated simultaneously:

| Domain | Type |
|---|---|
| 📱 SMS Smishing | Text-based binary classification |
| 📧 Email Phishing | Text-based binary classification |
| 💳 Financial Transaction Fraud | Numeric/behavioural binary classification |

---

## ⚠️ Problem Statement

Modern fraud detection systems suffer from **three structural failures**:

### Problem 1 — Concept Drift
Static models lose **15–25% accuracy within months** of deployment (Gama et al., 2014). Scam tactics mutate continuously — smishing evolves into AI voice cloning, phishing into deepfake video — but fixed models cannot adapt.

### Problem 2 — Data Quality & Class Imbalance
In real fraud datasets, fewer than **1% of records are genuine fraud**. Classifiers trained on such imbalanced data predict *legitimate* for virtually every transaction, achieving high accuracy while detecting almost no fraud.

### Problem 3 — Dynamic Scam Patterns
AI-driven fraud is projected to increase by **1,210% by 2025** (Cardiet, 2026). Fraudsters actively monitor detection triggers and pivot strategies once a pattern is flagged. Seasonal transaction surges are also frequently misidentified as concept drift, causing costly and unnecessary retraining.

---

## 💡 Proposed Solution

A four-component **integrated streaming pipeline**:

```
Incoming Data Stream
        │
        ▼
┌───────────────────────┐
│   C-SMOTE             │  ← Continuous class balancing (rolling reservoir)
│   Class Balancer      │
└──────────┬────────────┘
           │ balanced stream
           ▼
┌───────────────────────┐     ┌─────────────────────────┐
│   ARF                 │◄────│   ADWIN                 │
│   Core Classifier     │     │   Drift Detector        │
│   (10 trees, majority │     │   (adaptive sliding     │
│    vote)              │     │    window, δ = 0.002)   │
└──────────┬────────────┘     └──────────┬──────────────┘
           │                             │
           │              ┌──────────────▼──────────────┐
           │              │   False-Drift Analysis Step │
           │              │   Noise Filter              │
           │              │   (calendar-aware baseline) │
           │              └─────────────────────────────┘
           ▼
    Scam / Legitimate
      Prediction
```

| Component | Role | Key Property |
|---|---|---|
| **ARF** (Gomes et al., 2017) | Core streaming classifier | Tree-level replacement; no coverage gap |
| **ADWIN** (Bifet & Gavaldà, 2007) | Concept drift detector | Mathematical guarantees; <5% false alarm rate |
| **C-SMOTE** (Aguiar et al., 2023) | Streaming class balancer | Rolling reservoir; clears on confirmed drift |
| **False-Drift Analysis** | Noise filter | Suppresses seasonal spikes; preserves valid knowledge |

---

## 🎯 Research Objectives

| # | Research Question | Objective |
|---|---|---|
| RQ1 | What preprocessing pipeline structures scam datasets into a chronological, class-balanced, stream-ready format? | **(a)** Preprocess and structure datasets to simulate concept drift |
| RQ2 | How can an adaptive model combining ARF and ADWIN be designed to detect scams under concept drift? | **(b)** Design and develop the ARF + ADWIN detection model |
| RQ3 | To what extent does ARF + ADWIN + C-SMOTE outperform static baselines (RF, SVM)? | **(c)** Evaluate adaptive model against static baselines using 5 metrics |

---

## 🏗️ System Architecture

### Phase 1 — Data Collection & Preprocessing
```
Raw Dataset
    │
    ├── Remove nulls, duplicates & outliers (IQR on financial)
    ├── Standardise labels → binary  (scam = 1, legitimate = 0)
    ├── Sort records chronologically by timestamp
    ├── Feature extraction:
    │       SMS/Email  → TF-IDF (200/300 n-grams) + TruncatedSVD (50 dims)
    │       Financial  → Ordinal encoding + StandardScaler
    └── Apply C-SMOTE within streaming pipeline
```

### Phase 2 — Model Design, Development & Implementation
```
Adaptive Pipeline:
    ARF (River) + ADWIN (δ=0.002) + C-SMOTE (k=5, reservoir≥100) + False-Drift Filter

Static Baselines (Scikit-learn):
    Random Forest  |  Support Vector Machine (SVM)
```

### Phase 3 — Evaluation & Discussion
```
Prequential (test-then-train) evaluation on same stream
    → Precision, Recall, F1 at regular intervals
    → Adaptation Time & Recovery Time per drift event (ARF only)
    → Confusion matrices: 3 models × 3 datasets = 9 matrices
    → Comparative analysis: ARF vs RF vs SVM
```

---

## 📦 Datasets

| Fraud Domain | Dataset | Year | Source | Samples | Fraud Rate |
|---|---|---|---|---|---|
| 💳 Transaction Fraud | [Financial Transactions Dataset for Fraud Detection](https://www.kaggle.com/) — Kumar (2025) | 2025 | Kaggle | 5,000,000 | ~4% |
| 📧 Email Phishing | [Phishing Email Detection](https://www.kaggle.com/) — Chakraborty (2023) | 2023 | Kaggle | 18,650 | ~39% |
| 📱 SMS Smishing | [Balanced Dataset for Spam & Smishing Detection using LLMs](https://data.mendeley.com/) — Munoz & Islam (2025) | 2025 | Mendeley Data | 10,191 (raw) / 8,022 (cleaned) | ~42% |

> ⚠️ **Note:** The financial dataset (5M records) is not included in this repository due to file size. Please download it directly from Kaggle using the link above and place it in `data/raw/financial/`.

---

## 📁 Project Structure

```
adaptive-scam-detection/
│
├── data/
│   ├── raw/                        # Original downloaded datasets
│   │   ├── sms/
│   │   ├── email/
│   │   └── financial/
│   └── processed/                  # Cleaned, labelled, time-sorted datasets
│       ├── sms_cleaned.csv
│       ├── email_cleaned.csv
│       └── financial_cleaned.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA and visualisations (Figures 4.1–4.6)
│   ├── 02_preprocessing.ipynb      # Cleaning, labelling, TF-IDF, C-SMOTE
│   ├── 03_model_development.ipynb  # ARF + ADWIN + C-SMOTE pipeline (River)
│   └── 04_evaluation.ipynb         # Metrics, confusion matrices, comparison
│
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py         # NLTK tokenisation, stopword removal, stemming
│   │   ├── feature_extractor.py    # TF-IDF, TruncatedSVD, StandardScaler
│   │   └── csmote.py               # C-SMOTE streaming implementation
│   │
│   ├── models/
│   │   ├── adaptive_pipeline.py    # ARF + ADWIN + C-SMOTE + false-drift filter
│   │   ├── false_drift_filter.py   # Calendar-aware seasonal drift suppression
│   │   └── static_baselines.py     # Random Forest and SVM (Scikit-learn)
│   │
│   └── evaluation/
│       ├── metrics.py              # Precision, Recall, F1, Adaptation Time, Recovery Time
│       └── confusion_matrix.py     # Confusion matrix generation and plotting
│
├── results/                        # Output charts, confusion matrices, metric logs
│   ├── sms/
│   ├── email/
│   └── financial/
│
├── docs/
│   ├── thesis_PSM1.pdf             # Full PSM 1 thesis report
│   └── presentation_PSM1.pptx      # PSM 1 presentation slides
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/adaptive-scam-detection.git
cd adaptive-scam-detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
river==0.24.2
scikit-learn==1.8.0
nltk==3.9.4
pandas==2.3.3
numpy==2.4.6
matplotlib==3.10.9
seaborn==0.13.2
imbalanced-learn==0.14.1
```

### Download NLTK Resources

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### Quick Start

```python
from src.preprocessing.feature_extractor import extract_sms_features
from src.models.adaptive_pipeline import AdaptiveScamDetector

# Load and preprocess SMS dataset
X, y = extract_sms_features("data/processed/sms_cleaned.csv")

# Initialise adaptive pipeline
detector = AdaptiveScamDetector(
    n_trees=10,
    adwin_delta=0.002,
    csmote_k=5,
    csmote_min_reservoir=100
)

# Prequential evaluation (test-then-train)
for xi, yi in zip(X, y):
    prediction = detector.predict(xi)
    detector.learn(xi, yi)
```

---

## 🔬 Methodology

### Concept Drift Types Handled

| Type | Description | Example |
|---|---|---|
| **Abrupt** | Sudden, radical shift in fraud behaviour | Campaign pivot — abandon flagged tactic overnight |
| **Gradual** | Slow migration across channels over weeks | Investment fraud moving from email to messaging apps |
| **Incremental** | Progressive sophistication of an existing tactic | Smishing becoming personalised with names & account numbers |
| **Recurring** | Periodic seasonal patterns | Holiday shopping fraud spikes (FBI, 2025) |

### Why ARF over Static Random Forest?

```
Static RF:  Train once → Deploy → ❌ Miss up to 30% threats (Paldino et al., 2022)
                                  ❌ 15–25% accuracy drop in months (Gama et al., 2014)

ARF:        Train continuously → ADWIN detects drift → Background tree trains in parallel
                               → Weak tree replaced → ✅ No coverage gap
                               → 10–20% higher accuracy on fraud streams (Desta et al., 2026)
```

### Why C-SMOTE over Standard SMOTE?

| Property | Standard SMOTE | ADASYN | C-SMOTE |
|---|---|---|---|
| Data setting | Batch only | Batch only | ✅ Streaming |
| Oversampling trigger | Once before training | Once before training | ✅ Continuous |
| Concept drift handling | ❌ None | ❌ None | ✅ Reservoir clears on drift |
| ARF compatible | ❌ No | ❌ No | ✅ Yes |

### Key Parameters

| Method | Parameter | Value | Justification |
|---|---|---|---|
| ARF | `n_models` | 10 | Balance between stability and computation (Gomes et al., 2017) |
| ARF | `adwin_delta` (warning) | 0.001 | River default — triggers background tree training |
| ARF | `adwin_delta` (drift) | 0.001 | River default — triggers tree replacement |
| ARF | `grace_period` | 50 | Min samples before drift eligibility |
| ADWIN | `delta` | 0.002 | Classic default (Bifet & Gavaldà, 2007) |
| C-SMOTE | `k_neighbors` | 5 | Canonical SMOTE default (Chawla et al., 2002) |
| C-SMOTE | `min_size_minority` | 100 | Prevents low-quality early synthesis (Bernardo et al., 2020) |
| TF-IDF (SMS) | `max_features` | 200 | Top 200 informative n-grams |
| TF-IDF (Email) | `max_features` | 300 | Larger vocabulary for longer email text |
| TF-IDF (both) | `ngram_range` | (1, 2) | Captures single words + two-word phrases |

---

## 📊 Evaluation Metrics

| Metric | Formula | Why It Matters |
|---|---|---|
| **Precision** | TP / (TP + FP) | Fewer false alarms → less friction for legitimate users |
| **Recall** | TP / (TP + FN) | Fewer missed threats → less direct financial harm |
| **F1-Score** | 2 × (P × R) / (P + R) | Balanced measure for imbalanced datasets |
| **Adaptation Time** | Records from drift confirmation → replacement tree active | How quickly the system starts learning new patterns |
| **Recovery Time** | Records from replacement tree active → F1 restored to pre-drift level | How long the system stays in degraded state |

> **Why not just accuracy?**
> On the financial dataset (96% legitimate), a model that predicts *legitimate* for every record achieves **96% accuracy** while detecting **zero fraud**. Precision, Recall, and F1 expose this failure; accuracy hides it.

---

## ✅ PSM 1 Results

PSM 1 has fully completed **Research Objective (a)** — all three datasets are preprocessed and ready for adaptive model development.

### C-SMOTE Class Balancing Results

| Dataset | Before C-SMOTE | After C-SMOTE |
|---|---|---|
| SMS Smishing | Legit: 3,386 · Scam: 4,629 | Legit: **4,629** · Scam: **4,629** ✅ |
| Financial Transaction | Legit: 4,422,822 · Fraud: **164,726** ⚠️ | Legit: 4,422,822 · Fraud: **4,422,822** ✅ |
| Email Phishing | Safe: 10,978 · Phishing: 6,555 | Safe: **10,978** · Phishing: **10,978** ✅ |

### PSM 1 Deliverables Completed

- [x] All 3 datasets collected, cleaned, and binary-labelled
- [x] Every record sorted chronologically by timestamp
- [x] C-SMOTE applied to all 3 streaming pipelines
- [x] Literature review completed — 3-part research gap confirmed
- [x] Adaptive pipeline architecture finalised
- [x] All model parameters justified and documented

---

## 🗺️ PSM 2 Roadmap

- [ ] **Implement** ARF + ADWIN + C-SMOTE + False-Drift Filter pipeline in River
- [ ] **Implement** static Random Forest and SVM baselines in Scikit-learn
- [ ] **Evaluate** all models on Precision, Recall, F1, Adaptation Time, Recovery Time
- [ ] **Generate** 9 confusion matrices (3 models × 3 datasets)
- [ ] **Analyse** adaptation and recovery profiles per drift event
- [ ] **Compare** adaptive vs static performance as stream progresses

**Expected outcome:** The ARF + ADWIN + C-SMOTE system will demonstrate shorter Adaptation Time, shorter Recovery Time, and higher sustained F1-scores compared to static baselines across all three fraud domains.

---

## 🧰 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.14.3 | Primary programming language |
| River | 0.24.2 | Streaming ML — ARF and ADWIN |
| Scikit-learn | 1.8.0 | Static baselines (RF, SVM), TF-IDF, TruncatedSVD |
| NLTK | 3.9.4 | Text cleaning — tokenisation, stopwords, stemming |
| Pandas | 2.3.3 | Data loading, manipulation, cleaning |
| NumPy | 2.4.6 | Numerical computations |
| Matplotlib | 3.10.9 | Data visualisation |
| Seaborn | 0.13.2 | Statistical plots |
| imbalanced-learn | 0.14.1 | Offline SMOTE for baseline inspection |
| VS Code | 1.122.1 | Development environment |

---

## 👤 Author

**Ng Jin En**
Matric No: A23CS0146
Bachelor of Computer Science (Data Engineering)
Faculty of Computing, Universiti Teknologi Malaysia

**Supervisor:** Nurfazrina Binti Mohd Zamry

---
