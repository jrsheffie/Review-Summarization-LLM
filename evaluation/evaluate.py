"""
evaluate.py  [Phase 3 — scaffold]

Top-level evaluation script: loads generated summaries from both the
prompted LLM and fine-tuned BART, computes ROUGE + BERTScore against
reference summaries, and writes results to outputs/metrics/.

TODO (Phase 3): wire together rouge_metrics.py, bertscore_metrics.py, and
manual_evaluation.py once both models are producing summaries.
"""

from evaluation.bertscore_metrics import compute_bertscore
from evaluation.rouge_metrics import compute_rouge


def run_evaluation(predictions: list[str], references: list[str]) -> dict:
    rouge = compute_rouge(predictions, references)
    bertscore = compute_bertscore(predictions, references)
    return {"rouge": rouge, "bertscore": bertscore}


if __name__ == "__main__":
    raise NotImplementedError("Run once both models have generated summaries (Phase 3).")
