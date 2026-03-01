"""
Step 1: Initial Data Exploration
---------------------------------
Load Fake.csv and True.csv, and display:
  - Shape of both datasets
  - Column names
  - First 5 rows of each dataset
  - Basic info summary
"""

import pandas as pd

# ── Helpers ──────────────────────────────────────────────────────────────────


def load_datasets(fake_path: str, true_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fake and true news datasets from CSV files."""
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    return df_fake, df_true


def display_shape(df: pd.DataFrame, label: str) -> None:
    """Print the shape (rows, columns) of a DataFrame."""
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")


def display_columns(df: pd.DataFrame) -> None:
    """Print column names of a DataFrame."""
    print(f"  Columns: {list(df.columns)}")


def display_head(df: pd.DataFrame, n: int = 5, max_col_width: int = 80) -> None:
    """Display the first n rows, truncating long string columns for readability."""
    preview = df.head(n).copy()
    for col in preview.select_dtypes(include="object").columns:
        preview[col] = (
            preview[col]
            .str.slice(0, max_col_width)
            .apply(
                lambda x: (
                    x + "…" if isinstance(x, str) and len(x) == max_col_width else x
                )
            )
        )
    with pd.option_context(
        "display.max_colwidth", max_col_width + 5, "display.width", 200
    ):
        print(preview.to_string(index=False))


def display_info(df: pd.DataFrame) -> None:
    """Print the .info() summary of a DataFrame."""
    df.info()


def explore_dataset(df: pd.DataFrame, label: str) -> None:
    """Run all exploration steps for a single dataset."""
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  📰  {label}")
    print(f"{separator}\n")

    print("▸ Shape")
    display_shape(df, label)

    print("\n▸ Column Names")
    display_columns(df)

    print("\n▸ First 5 Rows")
    display_head(df)

    print("\n▸ Info Summary")
    display_info(df)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    fake_path = "data/Fake.csv"
    true_path = "data/True.csv"

    print("Loading datasets …")
    df_fake, df_true = load_datasets(fake_path, true_path)
    print("✔ Datasets loaded successfully.\n")

    explore_dataset(df_fake, "Fake News Dataset  (Fake.csv)")
    explore_dataset(df_true, "True News Dataset  (True.csv)")


if __name__ == "__main__":
    main()
