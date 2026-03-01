"""
Step 2: Label & Merge Datasets
-------------------------------
  - Add a binary 'label' column  →  0 = Fake, 1 = True
  - Concatenate both datasets into a single DataFrame
  - Shuffle and reset the index
  - Display the class distribution
"""

import pandas as pd

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_datasets(fake_path: str, true_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fake and true news CSV files."""
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    return df_fake, df_true


def add_labels(
    df_fake: pd.DataFrame, df_true: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a binary 'label' column: 0 for Fake, 1 for True."""
    df_fake["label"] = 0
    df_true["label"] = 1
    return df_fake, df_true


def merge_and_shuffle(df_fake: pd.DataFrame, df_true: pd.DataFrame) -> pd.DataFrame:
    """Concatenate both datasets, shuffle rows, and reset the index."""
    df_combined = pd.concat([df_fake, df_true], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_combined


def display_class_distribution(df: pd.DataFrame) -> None:
    """Print the class distribution with counts and percentages."""
    label_map = {0: "Fake", 1: "True"}
    counts = df["label"].value_counts().sort_index()

    print("▸ Class Distribution")
    print("-" * 35)
    for label_val, count in counts.items():
        pct = count / len(df) * 100
        print(
            f"  {label_map[label_val]:>5} (label={label_val}): {count:>6}  ({pct:.2f}%)"
        )
    print("-" * 35)
    print(f"  {'Total':>14}: {len(df):>6}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    fake_path = "data/Fake.csv"
    true_path = "data/True.csv"

    # 1. Load
    print("Loading datasets …")
    df_fake, df_true = load_datasets(fake_path, true_path)
    print("✔ Datasets loaded.\n")

    # 2. Label
    print("Adding binary labels …")
    df_fake, df_true = add_labels(df_fake, df_true)
    print(f"  Fake → label 0  ({len(df_fake)} rows)")
    print(f"  True → label 1  ({len(df_true)} rows)\n")

    # 3. Merge & Shuffle
    print("Concatenating and shuffling …")
    df = merge_and_shuffle(df_fake, df_true)
    print(f"  Combined shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 4. Quick sanity check
    print("▸ First 5 rows of merged dataset")
    print(df.head().to_string(index=True, max_colwidth=60))
    print()

    # 5. Class distribution
    display_class_distribution(df)


if __name__ == "__main__":
    main()
