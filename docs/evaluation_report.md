# Evaluation Report

## Method (per docs/methodology.md)

Per the task-granularity decision in `docs/methodology.md`, BART and the
prompted LLM solve different tasks and are evaluated separately against
their own appropriate ground truth:

- **BART** is fine-tuned and evaluated on single-review pairs (`Text` ->
  `Summary`), using the real `Summary` column from the Amazon Fine Food
  Reviews dataset. Metrics: ROUGE-1/2/L and BERTScore, computed on a
  held-out **test** subset (n=3,000, seed=42) that the model never saw
  during training or validation-loss monitoring.
- **The prompted LLM** produces product-level, multi-review synthesis with
  no reference summary available at that granularity, so it is evaluated
  via LLM-judge (Claude comparing the summary directly to the source
  reviews) and, eventually, the manual rubric.

## BART Results (n=3,000, held-out test set)

Fine-tuned with LoRA (r=16, alpha=32, dropout=0.05) on a 30k-row training
subset, 3 epochs (see `01_train_bart.ipynb`). Training loss: 2.61 -> 2.45 ->
2.58; validation loss: 2.996 -> 2.968 -> 2.953.

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| ROUGE-1 | 0.046 | 0.427 | 0.080 |
| ROUGE-2 | 0.014 | 0.159 | 0.025 |
| ROUGE-L | 0.042 | 0.397 | 0.072 |
| BERTScore | 0.815 | 0.871 | 0.842 |

### Why ROUGE is low but BERTScore is high: a length-mismatch, not a quality failure

The ROUGE precision/recall split reveals *why* the F1 scores look poor at
first glance. Recall is roughly 5-10x higher than precision across all
three ROUGE variants. This pattern is the signature of a **length mismatch**
between BART's generated summaries and the reference `Summary` values, not
a sign that BART's summaries are inaccurate:

- BART's fine-tuned output tends to be a full sentence or short paragraph,
  faithfully compressing the review's content.
- The dataset's `Summary` column, by contrast, is a **user-written review
  headline** -- often just a few words, and frequently stylistically terse
  or idiosyncratic rather than descriptive (e.g. "Flyboy Cessna" for a tea
  review, "Ultimate Blend!" for a coffee review -- see the qualitative
  examples in `01_train_bart.ipynb`'s sanity check).
- High recall means BART's longer output happens to contain most of the
  words that appear in the short reference. Low precision means BART's
  output also contains many additional words the terse reference never
  had -- inevitable when comparing a full-sentence summary against a
  headline-length target.

BERTScore F1 (0.842) tells a very different, more favorable story, because
it measures semantic similarity via contextual embeddings rather than exact
n-gram overlap -- it is far more tolerant of legitimate length and wording
differences between a compressive summary and a terse headline. Taken
together, **BART's fine-tuned summaries are semantically well-aligned with
the source reviews (per BERTScore), even though they do not lexically match
the dataset's headline-style reference summaries (per ROUGE)**.

This is a genuine limitation of using `Summary` as ground truth: it is a
reasonable training signal (review headlines do correlate with review
content) but an imperfect evaluation reference, since it was never written
to be a comprehensive summary in the first place. This caveat should be
kept in mind when interpreting the ROUGE numbers above, and is worth
noting explicitly rather than treating ROUGE F1 alone as a verdict on
summary quality.

## Prompted LLM: Preliminary LLM-Judge Results (n=3 products)

An early pipeline-validation run (see git history for the original n=3
smoke test) judged the prompted LLM's product-level summaries directly
against the source reviews, since no reference summary exists at that
granularity. All 3 cases favored the LLM's structured, multi-review
synthesis over a single unsynthesized excerpt. This sample is too small to
draw conclusions from and needs to be scaled up (see Next Steps).

## Next Steps

1. Fill in `evaluation/manual_evaluation.py` with real scores: sample
   ~10-15 products/reviews and rate both models (accuracy, completeness,
   readability, hallucination-freedom, overall quality) by hand, since this
   is the only fair head-to-head comparison per `docs/methodology.md` (the
   two models solve different-granularity tasks with different reference
   types, so no single automatic metric compares them directly).
2. Scale the LLM-judge run beyond n=3 (target: n=30-50 products).
3. Synthesize BART's automatic metrics + both models' manual rubric scores
   into the cross-model comparison called for in `docs/model_comparison.md`.
4. Track Claude API cost/latency at the larger scale for the feasibility
   analysis.
