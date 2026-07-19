"""
llm_prompting.py

Zero-shot summarization pipeline using a prompted LLM (Claude).
Takes a product's batch of reviews (as produced by data/preprocess.py ->
create_batches()) and returns a structured summary.
"""

from __future__ import annotations

import os
import anthropic

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

_client = None


def _get_client() -> anthropic.Anthropic:
    """Lazily create a single shared Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


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
    """Call the prompted LLM API on a single product's review batch."""
    config = config or LLMConfig()
    prompt = build_prompt(product_batch)

    client = _get_client()
    response = client.messages.create(
        model=getattr(config, "model", "claude-sonnet-5"),
        max_tokens=getattr(config, "max_tokens", 500),
        temperature=getattr(config, "temperature", 0.3),
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def summarize_products(product_batches: list[dict], config: LLMConfig | None = None) -> list[dict]:
    """
    Run summarize_product over a list of product batches.

    Returns a list of dicts: [{"product_id": ..., "summary": ...}, ...]
    Any individual failure is captured in the result rather than crashing
    the whole run.
    """
    config = config or LLMConfig()
    results = []
    for batch in product_batches:
        product_id = batch.get("product_id") or batch.get("asin") or batch.get("id")
        try:
            summary = summarize_product(batch, config=config)
        except Exception as e:
            summary = f"[ERROR] {e}"
        results.append({"product_id": product_id, "summary": summary})
    return results
