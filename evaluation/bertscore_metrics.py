"""
bertscore_metrics.py  [Phase 3 — scaffold]

Computes BERTScore (precision/recall/F1) between generated and reference
summaries, using contextual embeddings rather than surface n-gram overlap
(complements ROUGE, which is purely lexical).

TODO (Phase 3): pip install bert-score
"""

from __future__ import annotations


def compute_bertscore(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    """
    TODO: implement with the `bert-score` package:

        from bert_score import score
        P, R, F1 = score(predictions, references, lang=lang)
        return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}
    """
    raise NotImplementedError("Implement in Phase 3 once summaries are generated.")
