"""
model_runner.py

Single-command pipeline entry point: loads the preprocessed dataset, loads a
pretrained generative model, runs inference on a small batch of samples, and
saves representative outputs to outputs/samples/.

Usage:
    python src/model_runner.py
    python src/model_runner.py --model bart --n-samples 8
    python src/model_runner.py --model llm --n-samples 5

Supports two models (this project's core comparison, see docs/methodology.md):
    --model bart  Fine-tuned BART + LoRA, single-review summarization.
                  Falls back to the base pretrained facebook/bart-large-cnn
                  (zero-shot) if no fine-tuned checkpoint is found locally --
                  so this runs even on a fresh clone with no Colab training.
    --model llm   Prompted Claude summarization, multi-review product batches.
                  Requires ANTHROPIC_API_KEY to be set.

Falls back from processed data (data/processed/test.csv or
product_batches.json) to the small committed sample dataset
(data/raw/sample_reviews.csv) if the processed files aren't present --
so this also runs on a fresh clone that hasn't run the full data pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repo root is on the path regardless of how this script is
# invoked (e.g. `python src/model_runner.py` from the repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and save representative samples.")
    parser.add_argument("--model", choices=["bart", "llm"], default="bart",
                         help="Which model to run (default: bart -- runs locally, no API key needed)")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to generate (5-10 recommended)")
    parser.add_argument("--output-dir", default="outputs/samples", help="Directory to save generated samples")
    parser.add_argument("--checkpoint-dir", default="outputs/bart_lora_30k/final",
                         help="Path to a fine-tuned BART+LoRA checkpoint (bart mode only)")
    return parser.parse_args()


def load_review_data() -> pd.DataFrame:
    """Load review data for BART mode, preferring the full processed test
    set but falling back to the small committed sample dataset."""
    processed_path = REPO_ROOT / "data/processed/test.csv"
    sample_path = REPO_ROOT / "data/raw/sample_reviews.csv"

    if processed_path.exists():
        print(f"Loading processed test set from {processed_path}")
        df = pd.read_csv(processed_path, engine="python", on_bad_lines="warn")
    elif sample_path.exists():
        print(f"Processed data not found -- falling back to sample dataset at {sample_path}")
        df = pd.read_csv(sample_path, engine="python", on_bad_lines="warn")
    else:
        raise FileNotFoundError(
            "No review data found. Expected data/processed/test.csv (run "
            "notebooks/00_colab_setup.ipynb's download+preprocess flow) or "
            "data/raw/sample_reviews.csv (should be committed to the repo)."
        )

    df = df.dropna(subset=["Text", "Summary"])
    return df


def load_product_batches() -> list[dict]:
    """Load product review batches for LLM mode, building them on the fly
    from the sample dataset if the processed batches file isn't present."""
    batches_path = REPO_ROOT / "data/processed/product_batches.json"
    sample_path = REPO_ROOT / "data/raw/sample_reviews.csv"

    if batches_path.exists():
        print(f"Loading product batches from {batches_path}")
        with open(batches_path) as f:
            return json.load(f)

    if not sample_path.exists():
        raise FileNotFoundError(
            "No product batch data found. Expected data/processed/product_batches.json "
            "(run notebooks/00_colab_setup.ipynb's download+preprocess flow) or "
            "data/raw/sample_reviews.csv (should be committed to the repo)."
        )

    print(f"product_batches.json not found -- building batches on the fly from {sample_path}")
    from data.preprocess import (
        drop_missing_and_nulls,
        remove_duplicates,
        normalize_text_fields,
        filter_uninformative,
        filter_non_english,
        group_by_product,
        create_batches,
    )

    df = pd.read_csv(sample_path)
    df = drop_missing_and_nulls(df)
    df = remove_duplicates(df)
    df = normalize_text_fields(df)
    df = filter_uninformative(df, min_words=5)
    df = filter_non_english(df)
    df = group_by_product(df, min_reviews_per_product=1)  # relaxed for a small sample
    return create_batches(df, batch_size=10)


def run_bart(n_samples: int, checkpoint_dir: str) -> list[dict]:
    """Run BART inference on n_samples reviews, using a fine-tuned LoRA
    checkpoint if available, otherwise the base pretrained model."""
    import torch
    from transformers import BartTokenizer, BartForConditionalGeneration

    checkpoint_path = REPO_ROOT / checkpoint_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    if checkpoint_path.exists():
        print(f"Loading fine-tuned LoRA checkpoint from {checkpoint_path}")
        from peft import PeftModel
        tokenizer = BartTokenizer.from_pretrained(str(checkpoint_path))
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        model_label = "bart_finetuned"
    else:
        print(f"No fine-tuned checkpoint found at {checkpoint_path} -- using base facebook/bart-large-cnn (zero-shot)")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = base_model
        model_label = "bart_zeroshot"

    model.to(device)
    model.eval()

    df = load_review_data()
    sample = df.sample(n=min(n_samples, len(df)), random_state=42)

    results = []
    for _, row in sample.iterrows():
        try:
            inputs = tokenizer(str(row["Text"]), return_tensors="pt", max_length=1024, truncation=True).to(device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append({
                "model": model_label,
                "input": str(row["Text"]),
                "reference_summary": str(row.get("Summary", "")),
                "generated_summary": generated,
            })
        except Exception as e:
            print(f"Warning: generation failed for one sample, skipping. Error: {e}")

    return results


def run_llm(n_samples: int) -> list[dict]:
    """Run the prompted-LLM pipeline on n_samples product batches."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it as an environment variable "
            "before running --model llm (see README's API Keys section)."
        )

    from models.llm_prompting import summarize_products

    batches = load_product_batches()
    sample_batches = batches[:n_samples]

    results = []
    llm_results = summarize_products(sample_batches)
    for batch, llm_result in zip(sample_batches, llm_results):
        results.append({
            "model": "prompted_llm",
            "product_id": batch.get("product_id"),
            "num_reviews": batch.get("num_reviews"),
            "generated_summary": llm_result.get("summary") or llm_result.get("error"),
        })

    return results


def save_samples(results: list[dict], output_dir: str, model_name: str) -> None:
    out_path = REPO_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results, start=1):
        sample_path = out_path / f"sample_{i:02d}_{model_name}.txt"
        with open(sample_path, "w") as f:
            for key, value in result.items():
                f.write(f"{key.upper()}:\n{value}\n\n")

    manifest_path = out_path / f"manifest_{model_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    description_path = out_path / "README.md"
    with open(description_path, "a") as f:
        f.write(
            f"\n## {model_name} run -- {datetime.now().isoformat(timespec='seconds')}\n"
            f"- Samples generated: {len(results)}\n"
            f"- Files: sample_01_{model_name}.txt through sample_{len(results):02d}_{model_name}.txt\n"
            f"- Structured record: manifest_{model_name}.json\n"
        )

    print(f"Saved {len(results)} samples to {out_path}")


def main():
    args = parse_args()

    if args.model == "bart":
        results = run_bart(args.n_samples, args.checkpoint_dir)
        model_name = results[0]["model"] if results else "bart"
    else:
        results = run_llm(args.n_samples)
        model_name = "prompted_llm"

    if not results:
        print("No samples were generated -- check warnings above.")
        return

    save_samples(results, args.output_dir, model_name)


if __name__ == "__main__":
    main()
