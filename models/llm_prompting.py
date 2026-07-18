"""
llm_prompting.py  [Phase 2 — scaffold]

Zero-shot summarization pipeline using a prompted LLM (Claude or GPT-4).
Takes a product's batch of reviews (as produced by data/preprocess.py ->
create_batches()) and returns a structured summary.

This module is a scaffold: the prompt template and function signature are
set, but the actual API call needs your API key wired in during Phase 2.
"""

from __future__ import annotations

from models.config import LLMConfig

PROMPT_TEMPLATE = """You are an expert product analyst. Given the following customer reviews for a single product, write a structured summary.

Reviews:
{reviews_block}

Respond in exactly this format:

Overall sentiment: <one sentence>

Pros:
- <bullet point>
- <bullet point>

Cons:
- <bullet point>
- <bullet point>

Recommendation: <one sentence>
"""


def format_reviews_block(reviews: list[dict]) -> str:
    """Turn a batch's list of review dicts into numbered text for the prompt."""
    lines = []
    for i, r in enumerate(reviews, start=1):
        lines.append(f"{i}. ({r.get('rating', '?')} stars) {r.get('text', '')}")
    return "\n".join(lines)


def build_prompt(product_batch: dict) -> str:
    reviews_block = format_reviews_block(product_batch["reviews"])
    return PROMPT_TEMPLATE.format(reviews_block=reviews_block)


def summarize_product(product_batch: dict, config: LLMConfig | None = None) -> str:
    """
    Call the prompted LLM API on a single product's review batch.

    TODO (Phase 2): wire in the actual API call, e.g. via the Anthropic
    Python SDK:

        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": build_prompt(product_batch)}],
        )
        return response.content[0].text
    """
    config = config or LLMConfig()
    prompt = build_prompt(product_batch)
    raise NotImplementedError(
        "API call not yet wired in. Prompt built successfully:\n\n" + prompt
    )
