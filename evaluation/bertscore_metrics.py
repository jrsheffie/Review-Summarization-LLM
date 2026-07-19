"""
bertscore_metrics.py

Computes BERTScore (precision/recall/F1) between generated and reference
summaries, using contextual embeddings rather than surface n-gram overlap
(complements ROUGE, which is purely lexical).

Note: the first call downloads a RoBERTa-based scoring model from Hugging
Face (~1.4GB) -- this requires internet access and will be slow the first
time, then cached for subsequent calls. Not testable in a sandboxed
environment without Hugging Face access; verify this runs correctly in
Colab before relying on its output.
"""

from __future__ import annotations


def compute_bertscore(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    """Compute averaged BERTScore precision, recall, and F1.

    Args:
        predictions: model-generated summaries
        references: ground-truth summaries (same order/length as predictions)
        lang: language code, used to select the default scoring model

    Returns:
        dict like {"precision": .., "recall": .., "f1": ..}
    """
    if len(predictions) != len(references):
        raise ValueError(f"predictions ({len(predictions)}) and references ({len(references)}) must be same length")
    if len(predictions) == 0:
        raise ValueError("predictions and references must not be empty")

    from bert_score import score

    P, R, F1 = score(predictions, references, lang=lang, verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }
