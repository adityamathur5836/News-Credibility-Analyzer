"""
Step 13: Train Final Model & Save Artifacts
---------------------------------------------
  - Train the tuned Logistic Regression model on the full dataset
  - Save the fitted model and TF-IDF vectorizer using joblib
  - These .pkl files are loaded by app.py (the Streamlit app)
"""

import pandas as pd
import numpy as np
import string
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ── Data Loading & Preprocessing ─────────────────────────────────────────────

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

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    download_nltk_resources()

    # 1. Load and prepare data
    print("Loading datasets …")
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    df["title"] = df["title"].replace(r"^\s*$", np.nan, regex=True)
    df["text"]  = df["text"].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)

    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].str.strip()

    print(f"✔ Loaded {len(df)} articles\n")

    # 2. Preprocess
    print("Applying NLTK preprocessing …")
    df["clean_text"] = df["content"].apply(preprocess_text)
    print("✔ Preprocessing complete\n")

    # 3. Split
    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. TF-IDF Vectorization
    print("Fitting TF-IDF vectorizer (max_features=5000) …")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("✔ Vectorization complete\n")

    # 5. Train tuned Logistic Regression
    print("Training Logistic Regression (C=10, class_weight='balanced') …")
    model = LogisticRegression(C=10, class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("✔ Training complete\n")

    # 6. Quick evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"▸ Test Accuracy : {acc:.4f}")
    print(f"▸ Test F1 Score : {f1:.4f}\n")

    # 7. Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("✔ Saved model.pkl")
    print("✔ Saved vectorizer.pkl")

if __name__ == "__main__":
    main()
