"""
Step 3: Data Cleaning
---------------------
  - Remove duplicate rows based on the article text column
  - Check for missing values in 'title' and 'text'
  - Drop rows with missing critical content
  - Print the updated dataset shape
"""

import pandas as pd
import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_and_prepare() -> pd.DataFrame:
    """Load both CSVs, label, merge, and shuffle (reproduces Step 2)."""
    df_fake = pd.read_csv("data/Fake.csv")
    df_true = pd.read_csv("data/True.csv")

    df_fake["label"] = 0
    df_true["label"] = 1

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on the 'text' column."""
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    after = len(df)

    print("▸ Duplicate Removal (based on 'text' column)")
    print("-" * 45)
    print(f"  Rows before : {before:>6}")
    print(f"  Rows after  : {after:>6}")
    print(f"  Duplicates  : {before - after:>6}")
    print()
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """Report missing values in the 'title' and 'text' columns."""
    print("▸ Missing Values Check")
    print("-" * 45)
    for col in ["title", "text"]:
        null_count = df[col].isnull().sum()
        empty_count = (df[col].astype(str).str.strip() == "").sum()
        print(f"  {col!r:>8}  →  NaN: {null_count}  |  Empty strings: {empty_count}")
    print()


def drop_missing_critical(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'title' or 'text' is NaN or an empty/whitespace string."""
    before = len(df)

    # Replace empty / whitespace-only strings with NaN so dropna catches them
    df["title"] = df["title"].replace(r"^\s*$", np.nan, regex=True)
    df["text"] = df["text"].replace(r"^\s*$", np.nan, regex=True)

    df = df.dropna(subset=["title", "text"]).reset_index(drop=True)
    after = len(df)

    print("▸ Dropping Rows with Missing Critical Content (title / text)")
    print("-" * 45)
    print(f"  Rows before : {before:>6}")
    print(f"  Rows after  : {after:>6}")
    print(f"  Dropped     : {before - after:>6}")
    print()
    return df


def display_final_shape(df: pd.DataFrame) -> None:
    """Print the final shape of the cleaned dataset."""
    print("▸ Updated Dataset Shape")
    print("-" * 45)
    print(f"  {df.shape[0]} rows × {df.shape[1]} columns")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Load merged dataset (reproduces Step 2 pipeline)
    print("Loading and merging datasets …")
    df = load_and_prepare()
    print(f"✔ Merged dataset ready  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 2. Remove duplicates based on article text
    df = remove_duplicates(df)

    # 3. Check for missing values in title and text
    check_missing_values(df)

    # 4. Drop rows with missing critical content
    df = drop_missing_critical(df)

    # 5. Print updated dataset shape
    display_final_shape(df)

    # Quick sanity check
    print("▸ First 5 rows of cleaned dataset")
    print(df.head().to_string(index=True, max_colwidth=60))


if __name__ == "__main__":
    main()
