"""
manual_evaluation.py  [Phase 3 — scaffold]

Lightweight rubric for manually scoring a sample of generated summaries on:
accuracy, completeness, readability, hallucination presence, and overall
quality (1-5 scale each). Complements the automated ROUGE/BERTScore metrics
with a human-judgment check, since neither metric directly measures
factual accuracy against the source reviews.
"""

from dataclasses import dataclass


@dataclass
class ManualScore:
    product_id: str
    accuracy: int          # 1-5
    completeness: int      # 1-5
    readability: int       # 1-5
    hallucination_free: int  # 1-5 (5 = no hallucinations observed)
    overall_quality: int   # 1-5
    notes: str = ""


def save_scores(scores: list[ManualScore], path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ManualScore.__dataclass_fields__.keys()))
        writer.writeheader()
        for s in scores:
            writer.writerow(s.__dict__)
