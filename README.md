# Explainable AI (XAI) Tweets Project

This repository contains a notebook-based workflow for building, evaluating, and interpreting NLP models trained on tweet data. The project spans baseline modeling, transfer learning via a fine-tuned DistilBERT Named Entity Recognition (NER) model, and comprehensive Explainable AI (XAI) workflows.

---

## 📂 Project Structure

### Jupyter Notebooks
*   **[`Baseline Model.ipynb`](Baseline%20Model.ipynb):** Initial modeling experiments and baseline performance benchmarking.
*   **[`Transfer Learning.ipynb`](Transfer%20Learning.ipynb):** Workflow for fine-tuning and applying transfer learning to the tweet dataset.
*   **[`XAI.ipynb`](XAI.ipynb):** Explainability analysis, model interpretability, and feature attribution transitions.
*   **[`Evaluation.ipynb`](Evaluation.ipynb):** Performance evaluation and comparative analysis of model outputs against ground truth.

### Datasets & Data Files
*   **`tweets_cleaned.csv`:** Preprocessed and cleaned tweet dataset.
*   **`Tweets Trainset for Explainable AI.txt`:** Formatted training dataset.
*   **`Explainable AI Tweets Testset.txt`:** Formatted testing dataset.
*   **`Explainable AI Tweets Ground Truth.txt`:** Reference labels and ground truth annotations for evaluation.

### Model Artifacts
*   **`ner_distilbert_finetuned/`:** The final saved, fine-tuned DistilBERT NER model.
*   **`ner_distilbert_output/`:** Intermediate training checkpoints and evaluation logs.

---

## ⚙️ Core Pipeline Workflow

The workspace is designed as a sequential end-to-end NLP pipeline:

1.  **Data Preparation:** Cleaning and structuring raw tweet text.
2.  **Baseline Modeling:** Establishing a performance floor using standard ML techniques.
3.  **Transfer Learning:** Fine-tuning a DistilBERT model specifically for token classification/NER on tweets.
4.  **Explainability (XAI):** Extracting model insights to understand why specific predictions are made.
5.  **Evaluation:** Validating predictions against the official ground truth dataset.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment
This project requires Python 3.8+ along with core data science and deep learning libraries. It is highly recommended to use a virtual environment.

**Key Dependencies:**
*   **Data Manipulation & Viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`
*   **Machine Learning & NLP:** `scikit-learn`, `torch`, `transformers`, `datasets`
*   **Environment:** `jupyter` or VS Code Jupyter Extension

### 2. Installation
Clone the repository and install the required packages:

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install pandas numpy scikit-learn matplotlib seaborn torch transformers datasets jupyter