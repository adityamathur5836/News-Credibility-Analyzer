"""
Step 7: Stratified Train-Test Split
------------------------------------
  - Perform an 80-20 train-test split
  - Ensure the split is stratified based on the 'label' column
  - Print shapes of training and testing sets
  Uses only scikit-learn's model_selection module.
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

# ── Data Loading & Preprocessing (reproduces Steps 2–6) ──────────────────────

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

def load_and_preprocess_dataset(nrows: int = None) -> pd.DataFrame:
    """Load, label, merge, clean, add content, and apply NLTK preprocessing."""
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
    
    return df

# ── Train-Test Split ─────────────────────────────────────────────────────────

def perform_stratified_split(df: pd.DataFrame):
    """Perform an 80-20 stratified train-test split."""
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load and preprocess dataset (using a subset for faster verification)
    print("Loading and preprocessing dataset (subset of 5000 rows) …")
    df = load_and_preprocess_dataset(nrows=5000)
    print(f"✔ Dataset ready  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 2. Perform stratified split
    print("Performing 80-20 stratified train-test split …")
    X_train, X_test, y_train, y_test = perform_stratified_split(df)
    print("✔ Split complete\n")

    # 3. Print shapes
    print("▸ Dataset Shapes")
    print("-" * 45)
    print(f"  Training set (X_train): {X_train.shape}")
    print(f"  Testing set  (X_test) : {X_test.shape}")
    print(f"  Training labels (y_train): {y_train.shape}")
    print(f"  Testing labels  (y_test) : {y_test.shape}")
    print()

    # 4. Verify stratification
    print("▸ Stratification Check (Class Distribution)")
    print("-" * 45)
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    print("  Training Set:")
    for label, ratio in train_dist.items():
        print(f"    Label {label}: {ratio:.2%}")
        
    print("\n  Testing Set:")
    for label, ratio in test_dist.items():
        print(f"    Label {label}: {ratio:.2%}")

if __name__ == "__main__":
    main()
