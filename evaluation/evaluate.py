"""
evaluate.py

Top-level evaluation script with two modes, per docs/methodology.md's
task-granularity decision:

1. "reference" mode -- computes ROUGE + BERTScore for BART's single-review
   summaries against the dataset's real Summary column.

   Usage:
       python evaluation/evaluate.py reference \\
           --predictions outputs/summaries/bart_predictions.csv \\
           --pred-col prediction --ref-col Summary \\
           --output outputs/metrics/bart_results.json

2. "llm-judge" mode -- uses Claude to judge the prompted-LLM's product-level
   summaries directly against the original reviews, since no reference
   summary exists at that granularity.

   Usage:
       python evaluation/evaluate.py llm-judge \\
           --batches data/processed/product_batches.json \\
           --summaries outputs/summaries/llm_summaries.json \\
           --output outputs/metrics/llm_judge_results.json

The two modes are not comparable to each other numerically -- see
docs/methodology.md for why -- and are kept as separate code paths under
one entry point for convenience.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.bertscore_metrics import compute_bertscore
from evaluation.rouge_metrics import compute_rouge
from evaluation.llm_judge import judge_all


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


def evaluate_llm_judge(batches_path: str, summaries_path: str, output_path: str) -> list[dict]:
    with open(batches_path) as f:
        batches = json.load(f)
    with open(summaries_path) as f:
        llm_summaries = json.load(f)  # expected: {product_id: summary, ...}

    print(f"Judging {len(llm_summaries)} LLM summaries...")
    results = judge_all(batches, llm_summaries)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated summaries.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ref_parser = subparsers.add_parser("reference", help="ROUGE + BERTScore against a reference column (BART)")
    ref_parser.add_argument("--predictions", required=True, help="CSV with prediction and reference columns")
    ref_parser.add_argument("--pred-col", default="prediction", help="Column name with generated summaries")
    ref_parser.add_argument("--ref-col", default="reference", help="Column name with reference summaries")
    ref_parser.add_argument("--output", required=True, help="Path to save the JSON results")

    judge_parser = subparsers.add_parser("llm-judge", help="Claude-as-judge for prompted-LLM summaries (no reference)")
    judge_parser.add_argument("--batches", required=True, help="JSON file of product review batches")
    judge_parser.add_argument("--summaries", required=True, help="JSON file mapping product_id -> LLM summary")
    judge_parser.add_argument("--output", required=True, help="Path to save the JSON results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "reference":
        evaluate_from_csv(args.predictions, args.pred_col, args.ref_col, args.output)
    elif args.mode == "llm-judge":
        evaluate_llm_judge(args.batches, args.summaries, args.output)
