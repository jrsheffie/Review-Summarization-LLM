"""
rouge_metrics.py

Computes ROUGE-1, ROUGE-2, and ROUGE-L between generated summaries and
reference summaries.
"""

from __future__ import annotations

from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute averaged ROUGE-1/2/L precision, recall, and F1 across all
    prediction/reference pairs.

    Args:
        predictions: model-generated summaries
        references: ground-truth summaries (same order/length as predictions)

    Returns:
        dict like {"rouge1": {"precision": .., "recall": .., "fmeasure": ..}, "rouge2": {...}, "rougeL": {...}}
    """
    if len(predictions) != len(references):
        raise ValueError(f"predictions ({len(predictions)}) and references ({len(references)}) must be same length")
    if len(predictions) == 0:
        raise ValueError("predictions and references must not be empty")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]

    results = {}
    for metric in ["rouge1", "rouge2", "rougeL"]:
        precisions = [s[metric].precision for s in scores]
        recalls = [s[metric].recall for s in scores]
        fmeasures = [s[metric].fmeasure for s in scores]
        results[metric] = {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "fmeasure": sum(fmeasures) / len(fmeasures),
        }
    return results
