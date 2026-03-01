"""
Step 6: NLTK Text Preprocessing
--------------------------------
  - Convert text to lowercase
  - Remove punctuation
  - Remove stopwords
  - Perform lemmatization
  - Create 'clean_text' column from 'content'
  Uses only NLTK and standard Python libraries.
"""

import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── NLTK Downloads ───────────────────────────────────────────────────────────

def download_nltk_resources():
    """Download necessary NLTK datasets."""
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)

# ── Preprocessing Function ───────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Apply text preprocessing: lowercase, remove punctuation, remove stopwords, 
    and lemmatization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    # Using regex to remove punctuation for efficiency
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords and 5. Lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Filter stopwords and lemmatize in one pass
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join back into a single string
    return " ".join(cleaned_tokens)

# ── Data Loading (reproduces Steps 2–4) ─────────────────────────────────────

def load_dataset_with_content() -> pd.DataFrame:
    """Load, label, merge, clean, and add content column."""
    df_fake = pd.read_csv("data/Fake.csv")
    df_true = pd.read_csv("data/True.csv")

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
    return df

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Download NLTK resources
    print("Downloading NLTK resources …")
    download_nltk_resources()
    print("✔ Resources ready\n")

    # 2. Load dataset
    print("Loading dataset …")
    df = load_dataset_with_content()
    print(f"✔ Dataset ready  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 3. Apply preprocessing
    # Note: For large datasets, this might take a while. 
    # We'll apply it to the first 1000 rows for demonstration if it's too slow, 
    # but here we'll try the whole set or a significant chunk.
    print("Applying NLTK preprocessing to 'content' column …")
    print("(This may take a minute depending on dataset size)")
    
    # Applying to a subset for faster demonstration if needed, 
    # but the requirement is to apply it to the column.
    # To be safe and efficient, we use progress_apply if tqdm is available, 
    # but we'll stick to standard apply as per "standard Python libraries".
    df["clean_text"] = df["content"].apply(preprocess_text)
    
    print("✔ Preprocessing complete\n")

    # 4. Display results
    print("▸ Sample Results (content vs clean_text)")
    print("-" * 80)
    for idx, row in df.head(5).iterrows():
        print(f"Original [{idx}]: {row['content'][:100]}...")
        print(f"Cleaned  [{idx}]: {row['clean_text'][:100]}...")
        print("-" * 40)

    # 5. Show updated columns and shape
    print("\n▸ Updated Dataset Info")
    print("-" * 45)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Shape   : {df.shape[0]} rows × {df.shape[1]} columns")

if __name__ == "__main__":
    main()
