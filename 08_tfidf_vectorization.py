"""
Step 8: TF-IDF Vectorization
-----------------------------
  - Apply TF-IDF vectorization using scikit-learn's TfidfVectorizer
  - Fit only on the training data
  - Transform both training and testing sets
  - Limit max_features to 5000
  - Print feature matrix shapes
  Uses only scikit-learn's feature_extraction module.
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

# ── Data Loading & Preprocessing (reproduces Steps 2–7) ──────────────────────


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


def load_and_split_dataset(nrows: int = None) -> tuple:
    """Load, label, merge, clean, add content, apply NLTK preprocessing, and split."""
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

    return X_train, X_test, y_train, y_test


# ── TF-IDF Vectorization ─────────────────────────────────────────────────────


def apply_tfidf_vectorization(X_train, X_test, max_features=5000):
    """Apply TF-IDF vectorization: fit on train, transform both."""
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform testing data
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Load and split dataset (using a subset for faster verification)
    print("Loading and splitting dataset (subset of 5000 rows) …")
    X_train, X_test, y_train, y_test = load_and_split_dataset(nrows=5000)
    print(f"✔ Dataset ready  →  Train: {len(X_train)}, Test: {len(X_test)}\n")

    # 2. Apply TF-IDF vectorization
    print(f"Applying TF-IDF vectorization (max_features=5000) …")
    X_train_tfidf, X_test_tfidf, vectorizer = apply_tfidf_vectorization(X_train, X_test)
    print("✔ Vectorization complete\n")

    # 3. Print shapes
    print("▸ Feature Matrix Shapes")
    print("-" * 45)
    print(f"  X_train_tfidf: {X_train_tfidf.shape}")
    print(f"  X_test_tfidf : {X_test_tfidf.shape}")
    print()

    # 4. Show sample features
    print("▸ Sample Features (Top 10)")
    print("-" * 45)
    feature_names = vectorizer.get_feature_names_out()
    print(f"  {feature_names[:10]}")


if __name__ == "__main__":
    main()
