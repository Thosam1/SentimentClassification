
<div align="center">
  
# Leveraging LLM Ensembles for Robust Sentiment Classification

[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_SentimentClassification&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=Thosam1_SentimentClassification)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_SentimentClassification&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=Thosam1_SentimentClassification)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_SentimentClassification&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=Thosam1_SentimentClassification)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_SentimentClassification&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=Thosam1_SentimentClassification)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=Thosam1_SentimentClassification&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=Thosam1_SentimentClassification)

</div>

This repository contains the source code, data, models, and analysis for our sentiment classification project submitted for the **Computational Intelligence Lab (FS2025)** at ETH Zürich.

## 📝 Project Overview

We tackle **ternary sentiment classification** (positive, neutral, negative) using a dataset of 100,000+ sentences. Our approach compares classical ML models with state-of-the-art transformer-based architectures, explores preprocessing strategies, and evaluates ensembling techniques and LLM-generated paraphrasing.

Our best-performing model is a **softmax-averaged ensemble** of multiple fine-tuned transformer models:  
- `distilbert-base-multilingual-cased`  
- `deberta-v3-base`  
- `deberta-v3-large`  
- `roberta-large`  

This ensemble achieves:
- **L score**: 0.9034  
- **Weighted F1 score**: 0.83

For details, refer to our [📄 Project Report](./CIL_Sentiment_Analysis___Report.pdf).

---

## 📁 Folder Structure

```
.
├── config/               # Configuration scripts
├── data/                 # Raw training and test datasets
├── data_loader/          # Data loading logic
├── data_preprocessing/   # Preprocessing, language detection, LLM-based augmentation
├── fine_tuned_models/    # Checkpoints of fine-tuned transformer models
├── generated/            # Generated paraphrases, language info, misclassifications
├── models/               # Classical and transformer model definitions
├── notebooks/            # Jupyter notebooks for all experiments and analysis
├── scripts/              # Scripts to run jobs on cluster and to train LLM models
├── submissions/          # CSV submissions for the Kaggle competition
├── utils/                # Utility functions
├── visualizations/       # Plots and visual analysis functions
├── requirements.txt      # Project dependencies
└── README.md             # You're here!
```

---

## 🔧 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Thosam1/SentimentClassification.git
cd SentimentClassification
```

### 2. Create a virtual environment and activate it

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Notebooks

All experiments are reproducible through the provided Jupyter notebooks.

### Run notebooks locally

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` folder and open:

| Notebook | Purpose |
|----------|---------|
| `1_data_exploration.ipynb` | Dataset overview, class distribution |
| `2_basic_machine_learning.ipynb` | Classical ML models like Logistic Regression, RF, XGBoost |
| `3_large_language_models.ipynb` | Fine-tuning transformer models (BERT, RoBERTa, DeBERTa, etc.) |
| `4_roberta_analysis.ipynb` | Roberta-large misclassifications and error analysis |
| `5_inference_time_augmentation.ipynb` | LLM-generated paraphrasing for correcting misclassifications |
| `submission_models.ipynb` | Aggregation, ensembling (softmax & majority voting), submission generation |

---

## 🚀 Scripts for Training on a Cluster

To run fine-tuning jobs on a cluster:

```bash
# Bash entry point
bash scripts/batch.sh

# OR Python launcher
python scripts/run_job_on_cluster.py
```

---

## 🧠 Fine-Tuned Models

All fine-tuned model weights are stored under `fine_tuned_models/`. These include:

- BERT (base, multilingual, large)
- DistilBERT (base, multilingual)
- RoBERTa (base, large)
- DeBERTa v3 (base, large)
- XLM-RoBERTa (base)

Use these directly via Hugging Face’s `AutoModelForSequenceClassification`.

---

## 📊 Submissions

Submission files (CSV format) using various ensembling strategies are found in:

```
submissions/
├── deberta_large_submission.csv
├── majority_voting_submission.csv
└── softmax_averaging_submission.csv
```

---

## 📌 Future Work

- Test different classification heads and attention mechanisms
- Apply LLM translation for sanitizing input data
- Improve paraphrasing strategies using cheaper LLMs or distillation
- Apply selective LLM augmentation only to samples likely to be misclassified

---

## 👥 Authors

- Thösam Norlha-Tsang  
- Afonso Ferreira da Silva Domingues  
- Rahul Kaundal  
*Group: Siuuupremacy*  
**ETH Zürich — Computational Intelligence Lab FS2025**

