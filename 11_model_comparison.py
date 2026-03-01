"""
Step 11: Model Comparison — Logistic Regression vs Decision Tree
-----------------------------------------------------------------
  - Train both classifiers on the SAME TF-IDF train/test split
  - Evaluate using Accuracy, Precision, Recall, F1 Score
  - Print a side-by-side comparison table and both confusion matrices
  Classical ML only — no ensemble models or boosting.
"""

import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ── Data Loading & Preprocessing (reproduces Steps 2–8) ──────────────────────

def download_nltk_resources():
    """Download necessary NLTK datasets."""
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)

def preprocess_text(text: str) -> str:
    """Apply text preprocessing: lowercase, remove punctuation, remove stopwords, and lemmatization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def load_and_vectorize_dataset(nrows: int = None) -> tuple:
    """Load, label, merge, clean, add content, apply NLTK preprocessing, split, and vectorize."""
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)

    if nrows:
        df = df.sample(n=min(nrows, len(df)), random_state=42).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    df["title"] = df["title"].replace(r"^\s*$", np.nan, regex=True)
    df["text"]  = df["text"].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)

    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].str.strip()

    # Apply NLTK preprocessing
    download_nltk_resources()
    df["clean_text"] = df["content"].apply(preprocess_text)

    # Stratified Split
    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test

# ── Model Training ───────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Train a Decision Tree classifier."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics dict + predictions."""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    return metrics, y_pred

# ── Comparison Table ─────────────────────────────────────────────────────────

def print_comparison_table(lr_metrics: dict, dt_metrics: dict) -> None:
    """Print a formatted side-by-side comparison table."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    header = f"{'Metric':<12} | {'Logistic Regression':>20} | {'Decision Tree':>15}"
    separator = "-" * len(header)

    print(header)
    print(separator)

    for name in metric_names:
        lr_val = lr_metrics[name]
        dt_val = dt_metrics[name]
        print(f"  {name:<10} | {lr_val:>20.4f} | {dt_val:>15.4f}")

    print(separator)

def print_confusion_matrices(lr_cm, dt_cm) -> None:
    """Print both confusion matrices side by side."""
    print(f"\n  {'Logistic Regression':<25} {'Decision Tree'}")
    print(f"  [[TN: {lr_cm[0][0]:>4}, FP: {lr_cm[0][1]:>4}]     [[TN: {dt_cm[0][0]:>4}, FP: {dt_cm[0][1]:>4}]")
    print(f"   [FN: {lr_cm[1][0]:>4}, TP: {lr_cm[1][1]:>4}]]      [FN: {dt_cm[1][0]:>4}, TP: {dt_cm[1][1]:>4}]]")

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load and vectorize dataset
    print("Loading and vectorizing dataset (subset of 5000 rows) …")
    X_train_tfidf, X_test_tfidf, y_train, y_test = load_and_vectorize_dataset(nrows=5000)
    print(f"✔ Data ready  →  Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}\n")

    # 2. Train both models
    print("Training Logistic Regression …")
    lr_model = train_logistic_regression(X_train_tfidf, y_train)
    print("✔ Logistic Regression trained")

    print("Training Decision Tree …")
    dt_model = train_decision_tree(X_train_tfidf, y_train)
    print("✔ Decision Tree trained\n")

    # 3. Evaluate both models
    print("Evaluating both models on test set …")
    lr_metrics, lr_pred = evaluate_model(lr_model, X_test_tfidf, y_test)
    dt_metrics, dt_pred = evaluate_model(dt_model, X_test_tfidf, y_test)
    print("✔ Evaluation complete\n")

    # 4. Comparison Table
    print("=" * 55)
    print("  MODEL COMPARISON: Logistic Regression vs Decision Tree")
    print("=" * 55)
    print()
    print("▸ Performance Metrics")
    print("-" * 55)
    print_comparison_table(lr_metrics, dt_metrics)
    print()

    # 5. Confusion Matrices
    print("▸ Confusion Matrices")
    print("-" * 55)
    print_confusion_matrices(lr_metrics["Confusion Matrix"], dt_metrics["Confusion Matrix"])
    print()

    # 6. Individual Classification Reports
    print("▸ Classification Report — Logistic Regression")
    print("-" * 55)
    print(classification_report(y_test, lr_pred, target_names=["Fake", "True"]))

    print("▸ Classification Report — Decision Tree")
    print("-" * 55)
    print(classification_report(y_test, dt_pred, target_names=["Fake", "True"]))

    # 7. Summary
    print("=" * 55)
    lr_f1 = lr_metrics["F1 Score"]
    dt_f1 = dt_metrics["F1 Score"]
    if lr_f1 > dt_f1:
        winner = "Logistic Regression"
    elif dt_f1 > lr_f1:
        winner = "Decision Tree"
    else:
        winner = "Both models (tie)"
    print(f"  ★ Best model by F1 Score: {winner}")
    print("=" * 55)

if __name__ == "__main__":
    main()
