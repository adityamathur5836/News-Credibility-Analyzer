"""
Step 12: Hyperparameter Tuning & Feature Analysis
---------------------------------------------------
  - Tune Logistic Regression & Decision Tree via GridSearchCV
  - Re-evaluate tuned models with Accuracy, Precision, Recall, F1
  - Compare baseline vs tuned performance
  - If LR is the best tuned model, extract top influential words
  Uses only scikit-learn and numpy. No AutoML.
"""

import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Data Loading & Preprocessing (reproduces Steps 2–8) ──────────────────────


def download_nltk_resources():
    """Download necessary NLTK datasets."""
    resources = ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]
    for res in resources:
        nltk.download(res, quiet=True)


def preprocess_text(text: str) -> str:
    """Apply text preprocessing: lowercase, remove punctuation, remove stopwords, and lemmatization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    return " ".join(cleaned_tokens)


def load_and_vectorize_dataset(nrows: int = None) -> tuple:
    """Load, label, merge, clean, preprocess, split, and vectorize.
    Returns TF-IDF matrices, labels, AND the fitted vectorizer."""
    df_fake = pd.read_csv("data/Fake.csv")
    df_true = pd.read_csv("data/True.csv")

    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)

    if nrows:
        df = df.sample(n=min(nrows, len(df)), random_state=42).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    df["title"] = df["title"].replace(r"^\s*$", np.nan, regex=True)
    df["text"] = df["text"].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)

    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].str.strip()

    download_nltk_resources()
    df["clean_text"] = df["content"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# ── Evaluation Helper ────────────────────────────────────────────────────────


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate and return metrics dict + predictions."""
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics, y_pred


# ── Hyperparameter Tuning ────────────────────────────────────────────────────


def tune_logistic_regression(X_train, y_train):
    """Tune Logistic Regression with GridSearchCV."""
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"],
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_decision_tree(X_train, y_train):
    """Tune Decision Tree with GridSearchCV."""
    param_grid = {
        "max_depth": [10, 20, 50, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": [None, "balanced"],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ── Feature Analysis ─────────────────────────────────────────────────────────


def display_top_features(model, vectorizer, n_top: int = 15):
    """Extract and display top positive/negative LR coefficients."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    # Top positive coefficients → words indicating TRUE (label=1)
    top_positive_idx = np.argsort(coefficients)[-n_top:][::-1]
    # Top negative coefficients → words indicating FAKE (label=0)
    top_negative_idx = np.argsort(coefficients)[:n_top]

    print(f"\n  Top {n_top} words indicating TRUE news (positive coefficients):")
    print(f"  {'Rank':<6} {'Word':<20} {'Coefficient':>12}")
    print(f"  {'-'*6} {'-'*20} {'-'*12}")
    for rank, idx in enumerate(top_positive_idx, 1):
        print(f"  {rank:<6} {feature_names[idx]:<20} {coefficients[idx]:>12.4f}")

    print(f"\n  Top {n_top} words indicating FAKE news (negative coefficients):")
    print(f"  {'Rank':<6} {'Word':<20} {'Coefficient':>12}")
    print(f"  {'-'*6} {'-'*20} {'-'*12}")
    for rank, idx in enumerate(top_negative_idx, 1):
        print(f"  {rank:<6} {feature_names[idx]:<20} {coefficients[idx]:>12.4f}")


# ── Comparison Helpers ───────────────────────────────────────────────────────


def print_comparison_table(lr_metrics: dict, dt_metrics: dict) -> None:
    """Print tuned model comparison table."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    header = f"{'Metric':<12} | {'Logistic Regression':>20} | {'Decision Tree':>15}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for name in metric_names:
        print(f"  {name:<10} | {lr_metrics[name]:>20.4f} | {dt_metrics[name]:>15.4f}")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Load data
    print("Loading and vectorizing dataset (subset of 5000 rows) …")
    X_train, X_test, y_train, y_test, vectorizer = load_and_vectorize_dataset(
        nrows=5000
    )
    print(f"✔ Data ready  →  Train: {X_train.shape}, Test: {X_test.shape}\n")

    # ── 2. Baseline models (default hyperparameters) ─────────────────────────
    print("=" * 60)
    print("  PHASE 1: BASELINE MODELS (default hyperparameters)")
    print("=" * 60)

    lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
    lr_baseline.fit(X_train, y_train)
    lr_base_metrics, _ = evaluate_model(lr_baseline, X_test, y_test)

    dt_baseline = DecisionTreeClassifier(random_state=42)
    dt_baseline.fit(X_train, y_train)
    dt_base_metrics, _ = evaluate_model(dt_baseline, X_test, y_test)

    print("\n▸ Baseline Performance")
    print("-" * 60)
    print_comparison_table(lr_base_metrics, dt_base_metrics)
    print()

    # ── 3. Hyperparameter Tuning ─────────────────────────────────────────────
    print("=" * 60)
    print("  PHASE 2: HYPERPARAMETER TUNING (GridSearchCV, 5-fold CV)")
    print("=" * 60)

    print("\nTuning Logistic Regression …")
    lr_tuned, lr_best_params, lr_cv_score = tune_logistic_regression(X_train, y_train)
    print(f"  Best params : {lr_best_params}")
    print(f"  Best CV F1  : {lr_cv_score:.4f}")

    print("\nTuning Decision Tree …")
    dt_tuned, dt_best_params, dt_cv_score = tune_decision_tree(X_train, y_train)
    print(f"  Best params : {dt_best_params}")
    print(f"  Best CV F1  : {dt_cv_score:.4f}")
    print()

    # ── 4. Re-evaluate tuned models on test set ──────────────────────────────
    print("=" * 60)
    print("  PHASE 3: TUNED MODEL EVALUATION (test set)")
    print("=" * 60)

    lr_tuned_metrics, lr_pred = evaluate_model(lr_tuned, X_test, y_test)
    dt_tuned_metrics, dt_pred = evaluate_model(dt_tuned, X_test, y_test)

    print("\n▸ Tuned Performance")
    print("-" * 60)
    print_comparison_table(lr_tuned_metrics, dt_tuned_metrics)
    print()

    # ── 5. Baseline vs Tuned (per model) ─────────────────────────────────────
    print("▸ Improvement: Baseline → Tuned")
    print("-" * 60)
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    header = f"{'Metric':<12} | {'LR Δ':>10} | {'DT Δ':>10}"
    print(header)
    print("-" * len(header))
    for m in metric_names:
        lr_delta = lr_tuned_metrics[m] - lr_base_metrics[m]
        dt_delta = dt_tuned_metrics[m] - dt_base_metrics[m]
        print(f"  {m:<10} | {lr_delta:>+10.4f} | {dt_delta:>+10.4f}")
    print()

    # ── 6. Confusion matrices ────────────────────────────────────────────────
    print("▸ Confusion Matrices (Tuned)")
    print("-" * 60)
    lr_cm = lr_tuned_metrics["Confusion Matrix"]
    dt_cm = dt_tuned_metrics["Confusion Matrix"]
    print(f"\n  {'Logistic Regression':<25} {'Decision Tree'}")
    print(
        f"  [[TN: {lr_cm[0][0]:>4}, FP: {lr_cm[0][1]:>4}]     [[TN: {dt_cm[0][0]:>4}, FP: {dt_cm[0][1]:>4}]"
    )
    print(
        f"   [FN: {lr_cm[1][0]:>4}, TP: {lr_cm[1][1]:>4}]]      [FN: {dt_cm[1][0]:>4}, TP: {dt_cm[1][1]:>4}]]"
    )
    print()

    # ── 7. Classification Reports ────────────────────────────────────────────
    print("▸ Classification Report — Tuned Logistic Regression")
    print("-" * 60)
    print(classification_report(y_test, lr_pred, target_names=["Fake", "True"]))

    print("▸ Classification Report — Tuned Decision Tree")
    print("-" * 60)
    print(classification_report(y_test, dt_pred, target_names=["Fake", "True"]))

    # ── 8. Winner & Feature Analysis ─────────────────────────────────────────
    lr_f1 = lr_tuned_metrics["F1 Score"]
    dt_f1 = dt_tuned_metrics["F1 Score"]

    print("=" * 60)
    if lr_f1 >= dt_f1:
        print("  ★ Best tuned model by F1 Score: Logistic Regression")
        print("=" * 60)
        print("\n▸ Top Influential Words (from LR coefficients)")
        print("-" * 60)
        display_top_features(lr_tuned, vectorizer, n_top=15)
    else:
        print("  ★ Best tuned model by F1 Score: Decision Tree")
        print("=" * 60)
        print("\n  ℹ Decision Tree does not provide linear coefficients.")
        print("    Top influential words are available only for Logistic Regression.")

    print()


if __name__ == "__main__":
    main()
