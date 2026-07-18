"""
exploratory_analysis.py

Generates basic exploratory statistics and figures for the raw (or cleaned)
review dataset: review counts, rating distribution, review length
distribution, top product categories by review count, and missing-data
summary.

Usage:
    python data/exploratory_analysis.py \\
        --input data/raw/sample_reviews.csv \\
        --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe for Colab / CI
import matplotlib.pyplot as plt
import pandas as pd


def run_eda(input_path: str, output_dir: str) -> None:
    df = pd.read_csv(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Total reviews: {len(df)}")
    print(f"Unique products: {df['product_id'].nunique() if 'product_id' in df.columns else 'N/A'}")
    print("\nMissing values per column:")
    print(df.isna().sum())

    # Rating distribution
    if "rating" in df.columns:
        plt.figure(figsize=(6, 4))
        df["rating"].value_counts().sort_index().plot(kind="bar")
        plt.title("Rating Distribution")
        plt.xlabel("Star Rating")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out / "rating_distribution.png")
        plt.close()

    # Review length distribution
    if "review_text" in df.columns:
        lengths = df["review_text"].astype(str).str.split().str.len()
        plt.figure(figsize=(6, 4))
        lengths.plot(kind="hist", bins=30)
        plt.title("Review Length Distribution (words)")
        plt.xlabel("Word count")
        plt.tight_layout()
        plt.savefig(out / "review_length_distribution.png")
        plt.close()
        print(f"\nAverage review length: {lengths.mean():.1f} words")

    # Reviews per product (top products by volume)
    if "product_id" in df.columns:
        plt.figure(figsize=(6, 4))
        df["product_id"].value_counts().head(10).plot(kind="bar")
        plt.title("Top 10 Products by Review Count")
        plt.xlabel("Product ID")
        plt.ylabel("Number of Reviews")
        plt.tight_layout()
        plt.savefig(out / "top_products_by_review_count.png")
        plt.close()

    print(f"\nFigures saved to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis on the review dataset.")
    parser.add_argument("--input", required=True, help="Path to review CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to save figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(args.input, args.output_dir)
