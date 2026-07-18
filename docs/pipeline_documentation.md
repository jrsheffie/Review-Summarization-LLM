# Data Pipeline Documentation

## Overview

This document describes the preprocessing pipeline built for Milestone 3 of
*Summarizing Customer Product Reviews Using Generative AI*. The pipeline
takes the raw Amazon Product Reviews dataset and produces a cleaned,
deduplicated set of reviews, grouped by product ID, split into
train/validation/test sets for BART fine-tuning, and batched for prompting
the LLM.

## Input Schema

The pipeline expects a CSV with the following columns (see `COLUMN_MAP` in
`src/data_pipeline.py` if your downloaded version of the dataset uses
different column names):

| Column | Description |
|---|---|
| `product_id` | Unique identifier for the product being reviewed |
| `review_title` | Short title of the review |
| `review_text` | Full review text |
| `rating` | Star rating (1–5) |
| `helpful_votes` | Number of "helpful" votes the review received |

## Preprocessing Steps

1. **Drop missing/null rows** — reviews with no `product_id` or empty
   `review_text` are removed, since they can't be batched or summarized.
2. **Remove duplicates** — exact duplicate reviews (same product, title, and
   text) are dropped. Duplicate reviews are common in scraped review
   datasets and would otherwise bias both training and evaluation.
3. **Normalize text** — HTML tags are stripped and whitespace is collapsed.
4. **Filter uninformative reviews** — reviews under a minimum word count
   (default: 5 words) are removed, since one- or two-word reviews (e.g.
   "ok") don't carry summarizable content.
5. **Filter non-English reviews** — language is detected with `langdetect`
   (falling back to an ASCII-ratio heuristic if the library isn't
   available) and non-English reviews are dropped, since both the prompted
   LLM and fine-tuned BART model are being evaluated on English review
   summarization.
6. **Group by product** — products with fewer than a configurable minimum
   number of reviews (default: 2) are dropped, since summarizing across a
   single review isn't a meaningful test of the approach.

## Splitting Strategy

The train/validation/test split is done **by `product_id`, not by row**.
All reviews belonging to a given product are kept in the same split. This
prevents leakage: if some of a product's reviews were in training and
others in the test set, the fine-tuned BART model could effectively
memorize product-specific phrasing rather than learning to generalize.

Default split: 80% train / 10% validation / 10% test (by product count).

## Batching for the Prompted LLM

`create_batches()` groups cleaned reviews by `product_id` into a list of
dicts, capped at a configurable `batch_size` (default: 10 reviews per
product). This keeps prompt length manageable when passing multiple
reviews for a single product into the LLM API in one call. The output is
saved as `product_batches.json` and is the direct input to the Milestone 4
prompt templates.

## Edge Cases Handled

- Null or empty `review_text` / `product_id`
- Exact duplicate submissions
- HTML markup embedded in review text
- Very short, uninformative reviews (e.g. "ok", "")
- Non-English reviews
- Products with too few reviews to summarize meaningfully

## Outputs

Running `src/data_pipeline.py` produces, in `data/processed/`:

- `clean_reviews.csv` — full cleaned dataset
- `train.csv`, `val.csv`, `test.csv` — product-level splits for BART
- `product_batches.json` — reviews grouped by product for LLM prompting

## Getting the Real Dataset

1. Create a free Kaggle account and generate an API token
   (Account → Create New API Token → downloads `kaggle.json`).
2. Install the CLI: `pip install kaggle`
3. Place `kaggle.json` in `~/.kaggle/` (`chmod 600 ~/.kaggle/kaggle.json`)
4. Download the dataset, e.g.:
   ```bash
   kaggle datasets download -d <dataset-owner>/<dataset-slug> -p data/raw --unzip
   ```
5. Update `COLUMN_MAP` in `src/data_pipeline.py` if the downloaded column
   names differ from the schema above, then run:
   ```bash
   python src/data_pipeline.py --input data/raw/<file>.csv --output-dir data/processed
   ```
