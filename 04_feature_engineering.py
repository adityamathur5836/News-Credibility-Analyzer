"""
Step 4: Feature Engineering – Combine Title & Text
---------------------------------------------------
  - Create a new 'content' column by combining 'title' and 'text'
  - Handle NaN values safely using fillna
  - Display sample rows of the new column
"""

import pandas as pd
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_and_clean() -> pd.DataFrame:
    """Reproduce Steps 2–3: load, label, merge, deduplicate, drop missing."""
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove duplicates on text
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)

    # Replace empty / whitespace-only strings with NaN, then drop
    df["title"] = df["title"].replace(r"^\s*$", np.nan, regex=True)
    df["text"]  = df["text"].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)

    return df


def combine_title_and_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 'content' column by safely combining 'title' and 'text'."""
    # fillna("") ensures NaN values won't propagate into the combined string
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content"] = df["content"].str.strip()
    return df


def display_content_sample(df: pd.DataFrame, n: int = 5) -> None:
    """Print sample rows showing the new 'content' column."""
    print(f"▸ Sample Rows – 'content' column (first {n} rows)")
    print("-" * 80)
    for idx, row in df.head(n).iterrows():
        preview = row["content"][:150]
        suffix = " …" if len(row["content"]) > 150 else ""
        print(f"  [{idx}] (label={row['label']})  {preview}{suffix}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load cleaned dataset
    print("Loading and cleaning datasets …")
    df = load_and_clean()
    print(f"✔ Cleaned dataset ready  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 2. Combine title + text into 'content'
    print("Creating 'content' column (title + text) …")
    df = combine_title_and_text(df)
    print(f"✔ 'content' column added  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 3. Verify no NaN in the new column
    nan_count = df["content"].isnull().sum()
    print("▸ NaN Check on 'content' Column")
    print("-" * 45)
    print(f"  Missing values: {nan_count}")
    print()

    # 4. Display sample rows
    display_content_sample(df)

    # 5. Show updated columns and shape
    print("▸ Updated Dataset Info")
    print("-" * 45)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Shape   : {df.shape[0]} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()
