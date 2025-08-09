# spam_detector.py

import os
import requests
import zipfile
import io
import re
import string

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('punkt')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)
import joblib

sns.set(style="whitegrid")
RANDOM_STATE = 42

# ==== Step 1: Download dataset ====
print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

# Read into DataFrame
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print(f"Dataset loaded: {df.shape[0]} rows")

# ==== Step 2: Simple cleaning ====
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message'] = df['message'].apply(clean_text)

# ==== Step 3: Train/test split ====
X = df['clean_message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ==== Step 4: Build pipelines ====
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])

# ==== Step 5: Grid search for best Logistic Regression ====
param_grid_lr = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_df': [0.9, 1.0],
    'clf__C': [0.1, 1, 10]
}

print("Tuning Logistic Regression parameters...")
grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train, y_train)
print("Best params:", grid_lr.best_params_)

best_lr = grid_lr.best_estimator_

# ==== Step 6: Evaluate ====
y_pred = best_lr.predict(X_test)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred, target_names=['ham','spam']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham','spam'], yticklabels=['ham','spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - LogisticRegression")
plt.show()

# ROC AUC
if hasattr(best_lr.named_steps['clf'], "predict_proba"):
    prob_lr = best_lr.predict_proba(X_test)[:,1]
    auc_lr = roc_auc_score(y_test, prob_lr)
    print("ROC AUC:", auc_lr)

# ==== Step 7: Save model ====
joblib.dump(best_lr, "spam_classifier_lr.joblib")
print("Model saved as spam_classifier_lr.joblib")

# ==== Step 8: Predict on examples ====
examples = [
    "Congratulations! You have won a free ticket. Call now to claim.",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been compromised, reset password now."
]

loaded_model = joblib.load("spam_classifier_lr.joblib")
preds = loaded_model.predict(examples)
print("\nSample Predictions:")
for text, pred in zip(examples, preds):
    label = 'spam' if pred==1 else 'ham'
    print(f"{label.upper():>5} : {text}")
