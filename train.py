import os
import json
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib

from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from matplotlib.patches import FancyBboxPatch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'dataset.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# -----------------------------------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Loading Dataset")
print("=" * 70)

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} total SMS records.")
ham_count = int((df['label'] == 'ham').sum())
spam_count = int((df['label'] == 'spam').sum())
print(f"Ham  (Legitimate): {ham_count} ({ham_count / len(df) * 100:.2f}%)")
print(f"Spam (Fraudulent): {spam_count} ({spam_count / len(df) * 100:.2f}%)")

# -----------------------------------------------------------------------------
# 2. PREPROCESSING & CLEANING
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Text Preprocessing & Semantic Entity Masking")
print("=" * 70)

stemmer = PorterStemmer()
stopwords_set = set(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' httpaddr ', text)
    text = re.sub(r'\S+@\S+', ' emailaddr ', text)
    text = re.sub(r'[£$€₹]', ' moneysymb ', text)
    text = re.sub(r'\b\d{5,}\b', ' phonenum ', text)
    text = re.sub(r'\b\d+\b', ' numbr ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    cleaned = [stemmer.stem(w) for w in tokens if w not in stopwords_set and len(w) > 1]
    return " ".join(cleaned)

df['clean_message'] = df['message'].apply(preprocess_text)

# -----------------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT & TF-IDF VECTORIZATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Stratified Splitting & Sublinear TF-IDF Vectorization")
print("=" * 70)

X = df['clean_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train samples: {len(X_train)} (Ham: {(y_train==0).sum()}, Spam: {(y_train==1).sum()})")
print(f"Test samples : {len(X_test)} (Ham: {(y_test==0).sum()}, Spam: {(y_test==1).sum()})")

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
print(f"Vocabulary: {len(tfidf.vocabulary_)} features saved to models/tfidf_vectorizer.joblib")

# -----------------------------------------------------------------------------
# 4. MODEL TRAINING & EVALUATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Training & Evaluating Classifiers")
print("=" * 70)

models = {
    'Multinomial Naive Bayes': MultinomialNB(alpha=0.2),
    'Complement Naive Bayes': ComplementNB(alpha=0.2),
    'Bernoulli Naive Bayes': BernoulliNB(alpha=0.2),
    'Linear SVM (LinearSVC)': LinearSVC(C=1.0, random_state=42, max_iter=2000),
    'Support Vector Machine (RBF)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Logistic Regression (Baseline)': LogisticRegression(C=1.0, max_iter=1000, random_state=42)
}

results = {}
confusion_matrices = {}
y_probs = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    print(f"\nEvaluating {name}...")
    cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=skf, scoring='f1')
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X_test_tfidf)[:, 1]
    elif hasattr(clf, "decision_function"):
        df_score = clf.decision_function(X_test_tfidf)
        prob = (df_score - df_score.min()) / (df_score.max() - df_score.min())
    else:
        prob = y_pred
    y_probs[name] = prob
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    
    results[name] = {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1-Score': float(f1),
        'Specificity': float(spec),
        'ROC-AUC': float(roc_auc),
        'CV_F1_Mean': float(cv_scores.mean()),
        'CV_F1_Std': float(cv_scores.std()),
        'False_Positives': int(fp),
        'False_Negatives': int(fn)
    }
    print(f"  Accuracy : {acc*100:.2f}% | Precision: {prec*100:.2f}% (FP: {fp}) | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")

# Save primary models
joblib.dump(models['Multinomial Naive Bayes'], os.path.join(MODELS_DIR, 'multinomial_nb_model.joblib'))
joblib.dump(models['Complement Naive Bayes'], os.path.join(MODELS_DIR, 'complement_nb_model.joblib'))
joblib.dump(models['Linear SVM (LinearSVC)'], os.path.join(MODELS_DIR, 'linear_svm_model.joblib'))
joblib.dump(models['Support Vector Machine (RBF)'], os.path.join(MODELS_DIR, 'rbf_svm_model.joblib'))

# Save results
metrics_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC', 'False_Positives', 'False_Negatives']]
metrics_df_disp = metrics_df.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']:
    metrics_df_disp[col] = (metrics_df_disp[col] * 100).round(2)
metrics_df_disp['ROC-AUC'] = metrics_df_disp['ROC-AUC'].round(4)
metrics_df_disp.to_csv(os.path.join(RESULTS_DIR, 'model_metrics_table.csv'))

with open(os.path.join(RESULTS_DIR, 'model_evaluation_metrics.json'), 'w') as f:
    json.dump(results, f, indent=4)

print("\n" + "=" * 70)
print("FINAL PERFORMANCE SUMMARY TABLE:")
print(metrics_df_disp)
print("=" * 70)
print("All models successfully trained and artifacts saved to models/ and results/.")
