"""
Step 5: Exploratory Data Analysis – Visualizations
----------------------------------------------------
  - Plot class distribution (Fake vs True)
  - Plot histogram of article text length
  - Compare average text length per class
  Uses only matplotlib and seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import seaborn as sns


# ── Theme ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"Fake": "#e74c3c", "True": "#2ecc71"}


# ── Data Loading (reproduces Steps 2–4) ─────────────────────────────────────

def load_clean_dataset() -> pd.DataFrame:
    """Load, label, merge, clean, and add content column."""
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
    return df


# ── Plot 1: Class Distribution ──────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart showing Fake vs True article counts."""
    label_map = {0: "Fake", 1: "True"}
    df["class"] = df["label"].map(label_map)

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["class"].value_counts().reindex(["Fake", "True"])
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS["Fake"], COLORS["True"]],
                  edgecolor="white", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Class Distribution — Fake vs True", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.12)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig("plot_class_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✔ Saved → plot_class_distribution.png")


# ── Plot 2: Histogram of Article Text Length ────────────────────────────────

def plot_text_length_histogram(df: pd.DataFrame) -> None:
    """Histogram of article text lengths, coloured by class."""
    label_map = {0: "Fake", 1: "True"}
    df["class"] = df["label"].map(label_map)
    df["text_length"] = df["text"].str.len()

    fig, ax = plt.subplots(figsize=(9, 5))
    for cls in ["Fake", "True"]:
        subset = df[df["class"] == cls]
        ax.hist(subset["text_length"], bins=60, alpha=0.6,
                label=cls, color=COLORS[cls], edgecolor="white", linewidth=0.5)

    ax.set_title("Distribution of Article Text Length", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Text Length (characters)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(title="Class", fontsize=11, title_fontsize=11)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig("plot_text_length_histogram.png", dpi=150)
    plt.close(fig)
    print("  ✔ Saved → plot_text_length_histogram.png")


# ── Plot 3: Average Text Length per Class ───────────────────────────────────

def plot_avg_text_length(df: pd.DataFrame) -> None:
    """Bar chart comparing average text length for Fake vs True."""
    label_map = {0: "Fake", 1: "True"}
    df["class"] = df["label"].map(label_map)
    df["text_length"] = df["text"].str.len()

    avg_lengths = df.groupby("class")["text_length"].mean().reindex(["Fake", "True"])

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(avg_lengths.index, avg_lengths.values,
                  color=[COLORS["Fake"], COLORS["True"]],
                  edgecolor="white", linewidth=1.2, width=0.5)

    for bar, val in zip(bars, avg_lengths.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val:,.0f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_title("Average Article Text Length — Fake vs True",
                 fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Average Length (characters)", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylim(0, avg_lengths.max() * 1.15)
    sns.despine(left=True)
    fig.tight_layout()
    fig.savefig("plot_avg_text_length.png", dpi=150)
    plt.close(fig)
    print("  ✔ Saved → plot_avg_text_length.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading cleaned dataset …")
    df = load_clean_dataset()
    print(f"✔ Dataset ready  →  {df.shape[0]} rows × {df.shape[1]} columns\n")

    print("Generating visualizations …\n")

    plot_class_distribution(df)
    plot_text_length_histogram(df)
    plot_avg_text_length(df)

    print("\n✔ All plots saved successfully.")


if __name__ == "__main__":
    main()
