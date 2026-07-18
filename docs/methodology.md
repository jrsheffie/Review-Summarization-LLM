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

## Open Question to Resolve Before Phase 2

The Amazon Product Reviews dataset does not include reference/gold
summaries. Before fine-tuning BART, decide on one of:
1. Use review **titles** as a weak-supervision proxy for summaries
2. Generate synthetic reference summaries with the LLM and treat those as
   training targets for BART (introduces some circularity — worth noting
   as a limitation in the final report)
3. Use a different summarization-labeled dataset for BART's fine-tuning
   data, and Amazon Reviews only for the prompted-LLM comparison

Document whichever choice is made here, since it affects how the ROUGE/BERTScore
results should be interpreted.

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
