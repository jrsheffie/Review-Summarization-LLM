"""
Unit tests for src/data_pipeline.py

Run with:
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from preprocess import (  # noqa: E402
    clean_text,
    create_batches,
    drop_missing_and_nulls,
    filter_non_english,
    filter_uninformative,
    group_by_product,
    remove_duplicates,
    train_val_test_split,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["B001", "B001", "B001", "B002", "B002", "B003"],
        "review_title": ["Good", "Good", "Meh", "Nice", "Nice", None],
        "review_text": [
            "This product works really well and I would buy it again.",
            "This product works really well and I would buy it again.",  # exact duplicate
            "ok",  # too short
            "<b>Great</b> quality   for   the price, very happy overall.",
            None,  # null text
            "",  # empty text
        ],
        "rating": [5, 5, 3, 4, 4, 2],
        "helpful_votes": [10, 10, 0, 3, 3, 0],
    })


def test_drop_missing_and_nulls_removes_null_and_empty_text(sample_df):
    result = drop_missing_and_nulls(sample_df)
    assert result["review_text"].isna().sum() == 0
    assert (result["review_text"].str.strip() == "").sum() == 0
    assert len(result) == 4  # drops the None and the "" rows


def test_remove_duplicates_drops_exact_duplicate(sample_df):
    df = drop_missing_and_nulls(sample_df)
    result = remove_duplicates(df)
    # the two identical "Good" / same text rows for B001 collapse to one
    assert len(result[result["product_id"] == "B001"]) == 2  # "Good" (deduped) + "Meh"


def test_clean_text_strips_html_and_collapses_whitespace():
    dirty = "<b>Great</b>   quality   for   the   price"
    assert clean_text(dirty) == "Great quality for the price"


def test_filter_uninformative_removes_short_reviews():
    df = pd.DataFrame({"review_text": ["ok", "This is a sufficiently long review text"]})
    from preprocess import COLUMN_MAP
    df = df.rename(columns={"review_text": COLUMN_MAP["review_text"]})
    result = filter_uninformative(df, min_words=5)
    assert len(result) == 1
    assert "sufficiently" in result.iloc[0][COLUMN_MAP["review_text"]]


def test_filter_non_english_drops_non_english_text():
    df = pd.DataFrame({
        "product_id": ["B001", "B002"],
        "review_text": [
            "This is a perfectly normal English product review about quality.",
            "Ce produit est vraiment terrible et je ne le recommande pas du tout.",
        ],
        "review_title": ["a", "b"],
        "rating": [5, 1],
        "helpful_votes": [0, 0],
    })
    result = filter_non_english(df)
    assert len(result) == 1
    assert result.iloc[0]["product_id"] == "B001"


def test_group_by_product_filters_low_count_products():
    df = pd.DataFrame({
        "product_id": ["B001", "B001", "B002"],
        "review_text": ["a review here", "another review here", "lonely review here"],
    })
    result = group_by_product(df, min_reviews_per_product=2)
    assert set(result["product_id"].unique()) == {"B001"}


def test_create_batches_caps_batch_size():
    df = pd.DataFrame({
        "product_id": ["B001"] * 5,
        "review_title": ["t"] * 5,
        "review_text": ["some review text here"] * 5,
        "rating": [5] * 5,
        "helpful_votes": [0] * 5,
    })
    batches = create_batches(df, batch_size=3)
    assert len(batches) == 1
    assert batches[0]["num_reviews"] == 3


def test_train_val_test_split_keeps_products_together():
    df = pd.DataFrame({
        "product_id": [f"P{i}" for i in range(20) for _ in range(3)],
        "review_text": ["a review here"] * 60,
    })
    train_df, val_df, test_df = train_val_test_split(df, train_frac=0.7, val_frac=0.15)
    train_ids = set(train_df["product_id"])
    val_ids = set(val_df["product_id"])
    test_ids = set(test_df["product_id"])
    # no product should appear in more than one split
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    # all rows accounted for
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
