"""
evaluate.py

Compares BART baseline summaries against prompted-LLM summaries using:
1. ROUGE-L overlap between the two summaries (quantitative signal)
2. Claude-as-judge qualitative comparison, grounded in the original reviews
"""

from __future__ import annotations

import json
from rouge_score import rouge_scorer

from models.llm_prompting import _get_client, format_reviews_block
from models.config import LLMConfig

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_overlap(bart_summary: str, llm_summary: str) -> float:
    """ROUGE-L F1 between the two summaries — a rough lexical overlap signal,
    not a quality score. Low overlap is expected/fine since the LLM summary
    is meant to be more structured/abstractive."""
    scores = _scorer.score(bart_summary, llm_summary)
    return scores["rougeL"].fmeasure


JUDGE_PROMPT = """You are evaluating two AI-generated summaries of the same set of customer reviews for a product.

Original reviews:
{reviews_block}

Summary A (BART, extractive baseline):
{bart_summary}

Summary B (prompted LLM, structured):
{llm_summary}

Judge which summary better captures the key sentiment, pros, cons, and overall usefulness for a shopper deciding whether to buy this product. Respond in exactly this format:

Winner: <A, B, or Tie>
Reason: <one or two sentences>
"""


def judge_summaries(product_batch: dict, bart_summary: str, llm_summary: str,
                     config: LLMConfig | None = None) -> dict:
    """Use Claude to compare a BART summary against an LLM summary for one product."""
    config = config or LLMConfig()
    reviews_block = format_reviews_block(product_batch["reviews"])
    prompt = JUDGE_PROMPT.format(
        reviews_block=reviews_block,
        bart_summary=bart_summary,
        llm_summary=llm_summary,
    )

    client = _get_client()
    response = client.messages.create(
        model=config.model,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()

    winner = "Unknown"
    reason = text
    for line in text.splitlines():
        if line.lower().startswith("winner:"):
            winner = line.split(":", 1)[1].strip()
        if line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return {"winner": winner, "reason": reason, "raw": text}


def evaluate_all(product_batches: list[dict], bart_summaries: dict[str, str],
                  llm_summaries: dict[str, str], config: LLMConfig | None = None) -> list[dict]:
    """
    Run full evaluation across all products.

    bart_summaries / llm_summaries: dict mapping product_id -> summary string
    Returns a list of per-product result dicts.
    """
    config = config or LLMConfig()
    results = []
    for batch in product_batches:
        product_id = batch.get("product_id") or batch.get("asin") or batch.get("id")
        bart_summary = bart_summaries.get(product_id, "")
        llm_summary = llm_summaries.get(product_id, "")

        if not bart_summary or not llm_summary:
            results.append({
                "product_id": product_id,
                "error": "missing summary for one or both models",
            })
            continue

        overlap = rouge_l_overlap(bart_summary, llm_summary)
        judgment = judge_summaries(batch, bart_summary, llm_summary, config=config)

        results.append({
            "product_id": product_id,
            "bart_summary": bart_summary,
            "llm_summary": llm_summary,
            "rouge_l_overlap": round(overlap, 4),
            "judge_winner": judgment["winner"],
            "judge_reason": judgment["reason"],
        })
    return results
