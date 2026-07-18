"""
rouge_metrics.py  [Phase 3 — scaffold]

Computes ROUGE-1, ROUGE-2, and ROUGE-L between generated summaries and
reference summaries.

TODO (Phase 3): pip install rouge-score
"""

from __future__ import annotations


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """
    TODO: implement with the `rouge-score` package:

        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
        # then average precision/recall/fmeasure across `scores`
    """
    raise NotImplementedError("Implement in Phase 3 once summaries are generated.")
