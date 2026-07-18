"""
data_pipeline.py

Milestone 3 — Data Pipeline
Project: Summarizing Customer Product Reviews Using Generative AI

Cleans and filters the Amazon Product Reviews dataset, groups reviews by
product ID into batches, and produces train/validation/test splits suitable
for (a) prompting an LLM and (b) fine-tuning BART.

Usage:
    python src/data_pipeline.py \\
        --input data/raw/sample_reviews.csv \\
        --output-dir data/processed \\
        --min-words 5 \\
        --min-reviews-per-product 2 \\
        --batch-size 10
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

try:
    from langdetect import LangDetectException, detect
except ImportError:  # pragma: no cover
    detect = None
    LangDetectException = Exception

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------
# The public Kaggle "Amazon Product Reviews" dataset ships with slightly
# different column names depending on the version. Adjust this mapping once,
# here, rather than touching the functions below.
COLUMN_MAP = {
    "product_id": "ProductId",
    "review_title": "Summary",
    "review_text": "Text",
    "rating": "Score",
    "helpful_votes": "HelpfulnessNumerator",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw review CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------
def drop_missing_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a null/empty product_id or review_text."""
    before = len(df)
    df = df.copy()
    df[COLUMN_MAP["review_text"]] = df[COLUMN_MAP["review_text"]].astype("string")
    df = df.dropna(subset=[COLUMN_MAP["product_id"], COLUMN_MAP["review_text"]])
    df = df[df[COLUMN_MAP["review_text"]].str.strip() != ""]
    logger.info("drop_missing_and_nulls: %d -> %d rows", before, len(df))
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate reviews (same product, title, and text)."""
    before = len(df)
    subset = [COLUMN_MAP["product_id"], COLUMN_MAP["review_title"], COLUMN_MAP["review_text"]]
    df = df.drop_duplicates(subset=subset)
    logger.info("remove_duplicates: %d -> %d rows", before, len(df))
    return df


def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, and trim."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[COLUMN_MAP["review_text"]] = df[COLUMN_MAP["review_text"]].apply(clean_text)
    if COLUMN_MAP["review_title"] in df.columns:
        df[COLUMN_MAP["review_title"]] = df[COLUMN_MAP["review_title"]].fillna("").apply(clean_text)
    return df


def filter_uninformative(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    """Drop reviews that are too short to summarize meaningfully (e.g. "ok")."""
    before = len(df)
    word_counts = df[COLUMN_MAP["review_text"]].str.split().str.len()
    df = df[word_counts >= min_words]
    logger.info("filter_uninformative (min_words=%d): %d -> %d rows", min_words, before, len(df))
    return df


def _is_english(text: str) -> bool:
    if detect is None:
        # Fallback heuristic if langdetect isn't installed: mostly ASCII letters.
        letters = re.sub(r"[^A-Za-z]", "", text)
        return len(letters) / max(len(text), 1) > 0.6
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def filter_non_english(df: pd.DataFrame) -> pd.DataFrame:
    """Drop reviews not detected as English."""
    before = len(df)
    mask = df[COLUMN_MAP["review_text"]].apply(_is_english)
    df = df[mask]
    logger.info("filter_non_english: %d -> %d rows", before, len(df))
    return df


# ---------------------------------------------------------------------------
# Grouping and batching
# ---------------------------------------------------------------------------
def group_by_product(df: pd.DataFrame, min_reviews_per_product: int = 2) -> pd.DataFrame:
    """Keep only products that have at least `min_reviews_per_product` reviews,
    since a single review can't be meaningfully summarized across a product."""
    before_products = df[COLUMN_MAP["product_id"]].nunique()
    counts = df[COLUMN_MAP["product_id"]].value_counts()
    keep_ids = counts[counts >= min_reviews_per_product].index
    df = df[df[COLUMN_MAP["product_id"]].isin(keep_ids)]
    logger.info(
        "group_by_product (min=%d): %d -> %d products, %d rows",
        min_reviews_per_product, before_products, df[COLUMN_MAP["product_id"]].nunique(), len(df),
    )
    return df


def create_batches(df: pd.DataFrame, batch_size: int = 10) -> list[dict]:
    """Group cleaned reviews by product_id into batch dicts, ready to be
    passed either to an LLM prompt template or a BART fine-tuning example.
    Each batch caps the number of reviews per product at `batch_size` so
    prompts stay within a manageable context length."""
    batches = []
    for product_id, group in df.groupby(COLUMN_MAP["product_id"]):
        reviews = group.head(batch_size)
        batches.append({
            "product_id": product_id,
            "num_reviews": len(reviews),
            "reviews": [
                {
                    "title": row[COLUMN_MAP["review_title"]],
                    "text": row[COLUMN_MAP["review_text"]],
                    "rating": row[COLUMN_MAP["rating"]],
                    "helpful_votes": row.get(COLUMN_MAP["helpful_votes"], 0),
                }
                for _, row in reviews.iterrows()
            ],
        })
    logger.info("create_batches: %d product batches created", len(batches))
    return batches


# ---------------------------------------------------------------------------
# Splitting (for BART fine-tuning)
# ---------------------------------------------------------------------------
def train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by product_id (not by row) so all reviews for a given product
    stay in the same split — this avoids leaking a product's language/style
    between training and evaluation."""
    product_ids = df[COLUMN_MAP["product_id"]].drop_duplicates().sample(frac=1, random_state=seed)
    n = len(product_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(product_ids.iloc[:n_train])
    val_ids = set(product_ids.iloc[n_train:n_train + n_val])
    test_ids = set(product_ids.iloc[n_train + n_val:])

    train_df = df[df[COLUMN_MAP["product_id"]].isin(train_ids)]
    val_df = df[df[COLUMN_MAP["product_id"]].isin(val_ids)]
    test_df = df[df[COLUMN_MAP["product_id"]].isin(test_ids)]

    logger.info(
        "train_val_test_split: %d/%d/%d products -> %d/%d/%d rows",
        len(train_ids), len(val_ids), len(test_ids), len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    min_words: int = 5,
    min_reviews_per_product: int = 2,
    batch_size: int = 10,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df = drop_missing_and_nulls(df)
    df = remove_duplicates(df)
    df = normalize_text_fields(df)
    df = filter_uninformative(df, min_words=min_words)
    df = filter_non_english(df)
    df = group_by_product(df, min_reviews_per_product=min_reviews_per_product)

    clean_path = output_dir / "clean_reviews.csv"
    df.to_csv(clean_path, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows)", clean_path, len(df))

    train_df, val_df, test_df = train_val_test_split(df)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    batches = create_batches(df, batch_size=batch_size)
    batches_path = output_dir / "product_batches.json"
    with open(batches_path, "w") as f:
        json.dump(batches, f, indent=2)
    logger.info("Saved %d product batches to %s", len(batches), batches_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and batch the Amazon Reviews dataset.")
    parser.add_argument("--input", required=True, help="Path to raw reviews CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed outputs")
    parser.add_argument("--min-words", type=int, default=5, help="Minimum word count to keep a review")
    parser.add_argument("--min-reviews-per-product", type=int, default=2, help="Minimum reviews required per product")
    parser.add_argument("--batch-size", type=int, default=10, help="Max reviews per product batch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        min_words=args.min_words,
        min_reviews_per_product=args.min_reviews_per_product,
        batch_size=args.batch_size,
    )
