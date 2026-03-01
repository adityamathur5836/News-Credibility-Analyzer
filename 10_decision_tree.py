"""
Step 10: Decision Tree Training & Evaluation
----------------------------------------------
  - Train a Decision Tree classifier using TF-IDF features
  - Evaluate on the test set using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix
  Uses only scikit-learn metrics and models.
  No ensemble models or boosting — classical ML only.
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
    """Load, label, merge, clean, add content, apply NLTK preprocessing, split, and vectorize."""
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


# ── Model Training & Evaluation ──────────────────────────────────────────────


def train_decision_tree(X_train, y_train):
    """Train a Decision Tree classifier."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using various metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics, y_pred


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Load and vectorize dataset (using a subset for faster verification)
    print("Loading and vectorizing dataset (subset of 5000 rows) …")
    X_train_tfidf, X_test_tfidf, y_train, y_test = load_and_vectorize_dataset(
        nrows=5000
    )
    print(
        f"✔ Data ready  →  Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}\n"
    )

    # 2. Train Decision Tree
    print("Training Decision Tree classifier …")
    model = train_decision_tree(X_train_tfidf, y_train)
    print("✔ Training complete\n")

    # 3. Evaluate Model
    print("Evaluating model on test set …")
    metrics, y_pred = evaluate_model(model, X_test_tfidf, y_test)
    print("✔ Evaluation complete\n")

    # 4. Print Metrics
    print("▸ Performance Metrics")
    print("-" * 45)
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1 Score  : {metrics['F1 Score']:.4f}")
    print()

    print("▸ Confusion Matrix")
    print("-" * 45)
    cm = metrics["Confusion Matrix"]
    print(f"  [[TN: {cm[0][0]:>4}, FP: {cm[0][1]:>4}]")
    print(f"   [FN: {cm[1][0]:>4}, TP: {cm[1][1]:>4}]]")
    print()

    print("▸ Classification Report")
    print("-" * 45)
    print(classification_report(y_test, y_pred, target_names=["Fake", "True"]))


if __name__ == "__main__":
    main()
