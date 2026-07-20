# Evaluation Report

## Method (per docs/methodology.md)

Per the task-granularity decision in `docs/methodology.md`, BART and the
prompted LLM solve different tasks and are evaluated separately against
their own appropriate ground truth, then compared via a manual rubric:

- **BART** is fine-tuned and evaluated on single-review pairs (`Text` ->
  `Summary`), using the real `Summary` column from the Amazon Fine Food
  Reviews dataset. Automatic metrics: ROUGE-1/2/L and BERTScore, computed
  on a held-out **test** subset (n=3,000, seed=42) never seen during
  training or validation-loss monitoring.
- **The prompted LLM** produces product-level, multi-review synthesis with
  no reference summary available at that granularity, so it is evaluated
  via LLM-judge and the manual rubric instead.
- **Cross-model comparison** uses the manual rubric only (n=10 products,
  scored by hand), since no single automatic metric applies fairly to both
  models' different task granularities.

## BART: Automatic Metrics (n=3,000, held-out test set)

Fine-tuned with LoRA (r=16, alpha=32, dropout=0.05) on a 30k-row training
subset, 3 epochs (see `01_train_bart.ipynb`). Training loss: 2.61 -> 2.45 ->
2.58; validation loss: 2.996 -> 2.968 -> 2.953.

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| ROUGE-1 | 0.046 | 0.427 | 0.080 |
| ROUGE-2 | 0.014 | 0.159 | 0.025 |
| ROUGE-L | 0.042 | 0.397 | 0.072 |
| BERTScore | 0.815 | 0.871 | 0.842 |

**Why ROUGE is low but BERTScore is high:** ROUGE recall is roughly 5-10x
higher than precision across all three variants -- the signature of a
length mismatch, not inaccurate content. BART's fine-tuned output tends to
be a full sentence or short paragraph; the dataset's `Summary` column is a
user-written review headline, often just a few words and sometimes
stylistically unrelated to descriptive content (e.g. "Flyboy Cessna" for a
tea review). BERTScore's semantic-embedding approach is far more tolerant
of this length/wording gap, giving a more representative picture (F1 =
0.842) of how well BART's summaries actually align with review content.
This is a genuine limitation of `Summary` as an evaluation reference, worth
keeping in mind rather than treating ROUGE F1 alone as a verdict on quality.

## Manual Rubric: Cross-Model Comparison (n=10 products)

Ten products were sampled (seed=42) from the full dataset. For each: BART
summarized the single most-helpful review in the product's batch (BART's
task granularity), and the prompted LLM summarized the full multi-review
batch (its task granularity). Both were scored by hand on 1-5 scales for
accuracy, completeness, readability, hallucination-freedom, and overall
quality (see `evaluation/manual_evaluation.py`).

| Metric | BART | Prompted LLM |
|---|---|---|
| Accuracy | 2.60 | 5.00 |
| Completeness | 2.10 | 4.90 |
| Readability | 2.10 | 5.00 |
| Hallucination-free | 3.50 | 4.90 |
| Overall quality | 1.90 | 4.90 |

### Findings

The gap is large and consistent across every dimension, and it is
explainable rather than surprising:

- **BART, constrained to a single review with no cross-review synthesis,**
  frequently produced repetitive filler ("Love, love, love them!"),
  garbled or incomplete sentence fragments, and in several cases
  **fabricated details not present in its input** -- invented price points
  ($2.50, $5.00, $3.50, $4.50 on a product whose review only mentioned
  "$.175 each"), an invented "great price" claim, and an invented "chips"
  use case for a pickle relish. These are genuine hallucinations, not just
  stylistic weaknesses, and account for the model's low
  hallucination-free average (3.50) despite occurring on only a subset of
  the 10 samples.
- **One case (a peppermint-straws review) was a clear degenerate failure**:
  BART's output was almost entirely repeated exclamation marks, likely
  triggered by the source review's excessive "!!!" punctuation causing
  beam search to collapse into repetition. This is a notable qualitative
  limitation worth flagging on its own, separate from the aggregate scores.
- **The prompted LLM was consistently strong across all 10 products**,
  correctly synthesizing anywhere from 2 to 10 reviews per product into
  accurate, well-structured summaries -- including correctly representing
  *conflicting* reviews (e.g. a product with both a 5-star and a 1-star
  review was summarized with a balanced pros/cons split rather than
  favoring one side).
- This is not a simple "bigger model wins" result. It reflects the intended
  comparison from `docs/methodology.md`: BART's single-review task is
  inherently more fragile with limited context and only a 30k-row
  fine-tuning pass, while the LLM's multi-review synthesis task plays to
  zero-shot prompting's strength of aggregating and reasoning over more
  context at once. A BART model trained on the full ~317k rows for more
  epochs, or restructured to accept multi-review input, would likely close
  some of this gap -- see Next Steps.

## Prompted LLM: Preliminary LLM-Judge Results (n=3 products)

An earlier pipeline-validation run judged the LLM's summaries directly
against source reviews (no reference needed). All 3 cases favored the
LLM's structured synthesis over a single unsynthesized excerpt -- consistent
with the larger manual-rubric finding above, though n=3 is too small on its
own to draw conclusions from.

## Summary

The two approaches reflect the build-vs-buy tradeoff described in
`docs/model_comparison.md`: the prompted LLM required no training
investment and performed strongly out of the box on its natural task
(multi-review synthesis), while BART required a training pass and, even
after fine-tuning, showed real weaknesses on its single-review task at this
training scale -- including outright fabrication in some cases. This
suggests that for this specific product-review-summarization use case, at
this scale of fine-tuning, the zero-shot LLM approach currently produces
more reliable output, while BART's cost/reproducibility/independence
advantages (see `docs/model_comparison.md`) would need a larger or longer
training run to fully compete on quality.

## Next Steps

1. Consider training BART longer or on more data (the full ~317k rows, or
   more epochs) to test whether its hallucination and coherence issues
   diminish with additional fine-tuning.
2. Consider restructuring BART's input to accept multiple reviews per
   product (e.g. concatenation), narrowing the task-granularity gap between
   the two models for a more direct comparison.
3. Scale the LLM-judge run beyond n=3 (target: n=30-50 products) for a
   larger-sample confirmation of the manual rubric's findings.
4. Track Claude API cost/latency at that larger scale for the feasibility
   analysis in `docs/model_comparison.md`.
