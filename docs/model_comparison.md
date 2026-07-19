# Model Comparison / Benchmarking

This document compares the two candidate approaches — a prompted LLM
(Claude/GPT-4) and a LoRA-fine-tuned BART — across the factors relevant to
this project, as required by the milestone's benchmarking component.

## Framework Selection Justification

**Hugging Face Transformers + PEFT** were chosen for the BART side because:
- `facebook/bart-large-cnn` is available pretrained directly through the
  Transformers library, already fine-tuned for summarization (CNN/DailyMail),
  giving a strong starting point before task-specific LoRA fine-tuning.
- PEFT's LoRA implementation is well-maintained, integrates directly with
  Transformers' `Trainer`/`Seq2SeqTrainer`, and is the de facto standard for
  parameter-efficient fine-tuning at the time of writing.
- Both integrate natively with Colab's free GPU runtime.

**The Anthropic Python SDK** (or OpenAI's, as a fallback) was chosen for the
prompted-LLM side because it's the official, maintained client for the
model actually being evaluated — no reasonable alternative exists for
calling a hosted, closed-weight model.

## Comparison Table

| Factor | Prompted LLM (Claude/GPT-4) | Fine-tuned BART + LoRA |
|---|---|---|
| **Accuracy / output quality** | Strong general summarization ability out of the box; quality depends heavily on prompt design; no task-specific adaptation | Starts from a summarization-pretrained checkpoint; LoRA fine-tuning adapts it to this dataset's specific style and vocabulary |
| **Computational cost (inference)** | Pay-per-token API cost; no local compute needed; latency depends on provider | Runs locally/on Colab GPU once trained; effectively free per-inference after training, but requires GPU during training |
| **Computational cost (training)** | None — zero-shot, no training step at all | LoRA reduces trainable parameters substantially vs. full fine-tuning, keeping training feasible on a single Colab GPU (T4) |
| **Scalability (data volume)** | Scales with API rate limits and cost; summarizing all ~43,800 products would incur nontrivial per-call cost | Scales cheaply once trained — inference cost is just compute time, no per-call fee |
| **Scalability (task adaptation)** | Adapting to a new domain requires only prompt changes — fast to iterate | Adapting to a new domain requires re-running fine-tuning — slower to iterate but the model "owns" the pattern afterward |
| **Availability of pretrained models** | Only available via hosted API — no local weights, no offline use | Freely available on Hugging Face Hub; can be self-hosted, modified, or further fine-tuned without vendor dependency |
| **Task granularity** | Naturally handles multi-review batches in one call (see `docs/methodology.md`) | Naturally suited to single-document (single-review) summarization; multi-review synthesis would need a different framing (e.g. concatenation) |
| **Reproducibility** | Depends on provider's model version/availability over time (models can be deprecated) | Fully reproducible — same checkpoint + same LoRA weights produce same outputs indefinitely |

## Takeaway

The two approaches sit on opposite ends of a build-vs-buy tradeoff: the
prompted LLM requires no training investment and adapts instantly through
prompt iteration, at the cost of ongoing per-call expense and dependency on
a third-party API. The fine-tuned BART model requires an upfront training
cost (mitigated by LoRA) but then runs cheaply, reproducibly, and
independently of any external service. This is the central comparison the
project is designed to surface — not simply "which model writes a better
summary."
