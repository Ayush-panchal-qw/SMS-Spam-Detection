# SMS Spam Message Detection (Machine Learning Model)

**Topic:** 8. SMS Spam Message Detection Naive Bayes / SVM  
**Author:** Ayush Panchal  
**KUID:** KU2507U0431  
**Course:** Introduction to Machine Learning (IML)  
**Institution:** Unitedworld Institute of Technology (UIT), Karnavati University  

---

## 📁 Folder Contents

This folder contains **only the machine learning model files, dataset, training code, and evaluation results**:

- **`dataset.csv`**: The benchmark SMS Spam Collection dataset (5,574 labeled messages: 4,827 Ham, 747 Spam).
- **`train.py`**: Complete Python script to preprocess text, extract sublinear n-gram TF-IDF features, train 6 machine learning models (Multinomial NB, Bernoulli NB, Complement NB, Linear SVM, RBF SVM, Logistic Regression), evaluate cross-validation metrics, and serialize models.
- **`predict.py`**: Interactive prediction script to classify any custom SMS text in real time with confidence scores.
- **`SMS_Spam_Detection.ipynb`**: Clean Jupyter Notebook with detailed explanations and code cells for Viva demonstration.
- **`models/`**: Serialized pre-trained model binaries (`.joblib`):
  - `tfidf_vectorizer.joblib`
  - `linear_svm_model.joblib`
  - `multinomial_nb_model.joblib`
  - `complement_nb_model.joblib`
  - `rbf_svm_model.joblib`
- **`results/`**: Model evaluation charts and metrics:
  - `01_class_distribution.png`
  - `02_message_length_distribution.png`
  - `03_top_spam_words.png`
  - `04_confusion_matrices.png`
  - `05_model_comparison_bar.png`
  - `06_roc_curves.png`
  - `07_pipeline_architecture.png`
  - `model_metrics_table.csv`
  - `model_evaluation_metrics.json`
- **`requirements.txt`**: Python dependencies list.

---

## ⚡ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Custom SMS Messages Live
```bash
python predict.py
```

### 3. Retrain Models
```bash
python train.py
```

### 4. Run Jupyter Notebook
```bash
jupyter notebook SMS_Spam_Detection.ipynb
```

---

## 📊 Performance Summary (Test Set $N=1,115$)

| Model | Accuracy | Precision | Recall | F1-Score | Specificity | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (LinearSVC)** | **98.92%** | **98.58%** | **93.29%** | **95.86%** | **99.79%** | **2** |
| **Bernoulli Naive Bayes** | 99.01% | **100.0%** | 92.62% | 96.17% | **100.0%** | **0** |
| **Support Vector Machine (RBF)** | 98.74% | 99.27% | 91.28% | 95.10% | 99.90% | 1 |
| **Multinomial Naive Bayes** | 98.48% | 97.83% | 90.60% | 94.08% | 99.69% | 3 |
| **Logistic Regression (Baseline)** | 98.30% | 99.24% | 87.92% | 93.24% | 99.90% | 1 |
| **Complement Naive Bayes** | 97.40% | 86.14% | 95.97% | 90.79% | 97.62% | 23 |
