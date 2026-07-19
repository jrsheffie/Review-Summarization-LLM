# Evaluation Report

## Method (per docs/methodology.md)

Per the resolved task-granularity decision in `docs/methodology.md`, BART and
the prompted LLM solve different tasks and are evaluated separately against
their own ground truth, rather than against each other:

- **BART** is evaluated against the real `Summary` column from the Amazon
  Fine Food Reviews dataset (per-review reference summaries) using
  ROUGE-1/2/L and BERTScore.
- **The prompted LLM** produces product-level, multi-review synthesis with
  no reference summary available, so it is evaluated using the manual
  rubric (`evaluation/manual_evaluation.py`) and an LLM-judge comparison
  against the original review text directly, not against BART's output.
- The **cross-model comparison** is qualitative, via the manual rubric,
  reflecting the zero-shot vs. fine-tuned tradeoff rather than a shared
  numeric metric.

## Status: Preliminary / Pipeline Validation

BART fine-tuning has not yet been run (`training/train_bart.py` is still a
scaffold — see Roadmap in `README.md`), so ROUGE/BERTScore results against
the `Summary` ground truth are **not yet available**. This section will be
completed once Phase 2 training is done.

What follows is a preliminary validation of the **prompted-LLM evaluation
pipeline only** (`models/llm_prompting.py` + LLM-judge), run on a small
sample (n=3) to confirm the pipeline works end-to-end before scaling.

### Preliminary LLM-Judge Results (n=3 products)

An early prototype additionally computed ROUGE-L directly between BART's
zero-shot (not fine-tuned) output and the LLM's output, for pipeline-testing
purposes only. Per the task-granularity reasoning above, that number is
**not a meaningful evaluation metric** (the two models aren't solving the
same task) and is omitted here to avoid implying otherwise. See git history
for the earlier draft if needed for reference.

| Product ID | LLM-Judge Winner | Judge Reasoning (summarized) |
|---|---|---|
| 0006641040 | Prompted LLM | Better synthesis across all reviews; captured both praise for content and the recurring size/quality complaint that a single extracted sentence misses |
| 2734888454 | Prompted LLM | Structured output separated universal praise from the country-of-origin concern more clearly than an unsynthesized excerpt |
| 7310172001 | Prompted LLM | Synthesized all 10 reviews into a structured overview versus a single reproduced review |

### Observations

- The LLM-judge pipeline runs correctly end-to-end and produces reasoned,
  specific justifications rather than generic preferences.
- In this small sample, the prompted LLM's structured, multi-review
  synthesis was consistently judged more useful to a prospective buyer
  than BART's current zero-shot output, which tends to reproduce 1-2
  source sentences with limited cross-review synthesis. This is expected:
  the BART model being tested here is not yet fine-tuned on this dataset's
  `Summary` targets, so this is not yet a fair fine-tuned-vs-zero-shot
  comparison — it's closer to zero-shot-vs-zero-shot until training runs.
- This sample size (n=3) is too small to draw conclusions from — it only
  validates that the pipeline runs correctly.

## Next Steps

1. Complete BART + LoRA fine-tuning (`training/train_bart.py`) against the
   real `Summary` column.
2. Run ROUGE-1/2/L and BERTScore for fine-tuned BART against held-out
   `Summary` values (see `evaluation/rouge_metrics.py`,
   `evaluation/bertscore_metrics.py`).
3. Run the manual rubric (`evaluation/manual_evaluation.py`) on a larger,
   randomly sampled set of products (target: n=30-50) for both models.
4. Produce the qualitative cross-model comparison required by
   `docs/methodology.md`, once both models have real, comparable outputs
   to assess side by side.
5. Track Claude API cost/latency at the larger scale for the feasibility
   analysis in `docs/model_comparison.md`.
