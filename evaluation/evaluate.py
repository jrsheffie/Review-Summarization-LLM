"""
evaluate.py

Top-level evaluation script: computes ROUGE + BERTScore for a set of
generated summaries against reference summaries, and saves results to
outputs/metrics/.

Usage (BART, single-review summaries vs. real Summary column):
    python evaluation/evaluate.py \\
        --predictions outputs/summaries/bart_predictions.csv \\
        --pred-col prediction --ref-col Summary \\
        --output outputs/metrics/bart_results.json

The predictions CSV must have one column with model outputs and one column
with reference summaries, same row order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.bertscore_metrics import compute_bertscore
from evaluation.rouge_metrics import compute_rouge


def run_evaluation(predictions: list[str], references: list[str]) -> dict:
    """Compute both ROUGE and BERTScore for a list of predictions/references."""
    rouge = compute_rouge(predictions, references)
    bertscore = compute_bertscore(predictions, references)
    return {"rouge": rouge, "bertscore": bertscore}


def evaluate_from_csv(predictions_path: str, pred_col: str, ref_col: str, output_path: str) -> dict:
    df = pd.read_csv(predictions_path)
    df = df.dropna(subset=[pred_col, ref_col])

    predictions = df[pred_col].astype(str).tolist()
    references = df[ref_col].astype(str).tolist()

    print(f"Evaluating {len(predictions)} prediction/reference pairs...")
    results = run_evaluation(predictions, references)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated summaries against references.")
    parser.add_argument("--predictions", required=True, help="CSV with prediction and reference columns")
    parser.add_argument("--pred-col", default="prediction", help="Column name with generated summaries")
    parser.add_argument("--ref-col", default="reference", help="Column name with reference summaries")
    parser.add_argument("--output", required=True, help="Path to save the JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_from_csv(args.predictions, args.pred_col, args.ref_col, args.output)
