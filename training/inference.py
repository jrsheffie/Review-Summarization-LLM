"""
inference.py  [Phase 2 — scaffold]

Generates summaries from a fine-tuned BART checkpoint using beam search
(favoring coherent, high-likelihood output, per the proposal's decoding
strategy -- in contrast to the prompted LLM's default sampling).

TODO (Phase 2): load checkpoint, run model.generate(num_beams=4, ...)
"""


def generate_summary(review_text: str, model, tokenizer, num_beams: int = 4) -> str:
    raise NotImplementedError("Implement after BART fine-tuning in Phase 2.")
