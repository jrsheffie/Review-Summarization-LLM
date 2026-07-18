# Methodology

## Problem

Summarize customer product reviews (Amazon Product Reviews dataset) using
two generative approaches, compared directly against each other.

## Approach 1 — Prompted LLM (zero-shot)

- Model: Claude or GPT-4, accessed via API (decoder-only, autoregressive)
- No task-specific training — relies entirely on prompt engineering
- Reviews for a product are batched (see `data/preprocess.py:create_batches`)
  and passed in a single structured prompt (see `models/llm_prompting.py`)
- Output format is fixed: overall sentiment, pros, cons, recommendation —
  chosen specifically so outputs are directly comparable to BART's

## Approach 2 — Fine-tuned BART + LoRA

- Base model: `facebook/bart-large-cnn` (bidirectional encoder + autoregressive
  decoder, pretrained with a denoising objective)
- Fine-tuned with LoRA (rank=16, alpha=32, dropout=0.05) rather than full
  fine-tuning, to stay feasible on a single Colab GPU
- Decoding at inference uses beam search (favoring coherent, high-likelihood
  output over the more exploratory decoding typical of prompted LLMs)

## Resolved: Training Targets and Task Granularity (decided after inspecting real data)

The dataset in use is Amazon Fine Food Reviews (`Reviews.csv`), which
includes a `Summary` column — a short, human-written headline for each
**individual** review (e.g. "great tea", "awesome!"). It is not a synthesis
across multiple reviews for a product.

This resolves the earlier open question, but with an important caveat:
**the two models operate at different granularities, by design:**

- **BART** is fine-tuned on single-review pairs: `Text` (input) → `Summary`
  (target). This is a standard, well-suited task for BART's architecture.
- **The prompted LLM** operates on multi-review batches per product (see
  `data/preprocess.py:create_batches`), producing a structured summary
  (sentiment / pros / cons / recommendation) synthesized across several
  reviews at once.

**Implication for evaluation:** ROUGE and BERTScore should be computed
*within* each model's own task (BART vs. held-out `Summary` values; LLM vs.
a small manually-curated or LLM-assisted set of product-level reference
summaries) rather than directly against each other. The cross-model
comparison in the final report should lean on the manual rubric
(`evaluation/manual_evaluation.py`) for a fair head-to-head judgment, since
the two models are not solving numerically identical tasks — the
comparison is about the zero-shot vs. fine-tuned tradeoff, not about
matching output format.

## Evaluation

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — lexical overlap with reference summaries
- **BERTScore** (precision/recall/F1) — semantic similarity via contextual embeddings
- **Manual rubric** — accuracy, completeness, readability, hallucination-freedom,
  overall quality (1–5 each), scored by hand on a sample of outputs

## Comparison Framing

The central comparison is the tradeoff between:
- A **zero-shot, general-purpose** model (no training cost, relies on prompt
  quality)
- A **lightly fine-tuned, task-specific** model (training cost via LoRA, but
  learns domain-specific patterns directly from the data)
