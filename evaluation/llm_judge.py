"""
llm_judge.py

Evaluates the prompted-LLM's product-level summaries, which have no
reference summary to compare against (see docs/methodology.md's
task-granularity discussion). Uses Claude as a judge, comparing the LLM's
summary directly against the original reviews for accuracy, completeness,
and usefulness.

This is intentionally separate from rouge_metrics.py / bertscore_metrics.py,
which require a reference summary and are used for BART's evaluation
instead. Per docs/methodology.md, the two models are not compared via a
shared numeric metric -- this module and the manual rubric
(manual_evaluation.py) are how the LLM's output is assessed.
"""

from __future__ import annotations

from models.llm_prompting import _get_client, format_reviews_block
from models.config import LLMConfig

JUDGE_PROMPT = """You are evaluating an AI-generated summary of customer reviews for a product.

Original reviews:
{reviews_block}

Generated summary:
{summary}

Rate the summary's accuracy, completeness, and usefulness for a shopper deciding whether to buy this product. Respond in exactly this format:

Accuracy (1-5): <score>
Completeness (1-5): <score>
Usefulness (1-5): <score>
Notes: <one or two sentences>
"""


def judge_summary(product_batch: dict, summary: str, config: LLMConfig | None = None) -> dict:
    """Use Claude to rate a single product's LLM-generated summary against
    its original reviews."""
    config = config or LLMConfig()
    reviews_block = format_reviews_block(product_batch["reviews"])
    prompt = JUDGE_PROMPT.format(reviews_block=reviews_block, summary=summary)

    client = _get_client()
    response = client.messages.create(
        model=config.model,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()

    scores = {"accuracy": None, "completeness": None, "usefulness": None, "notes": ""}
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("accuracy"):
            scores["accuracy"] = _extract_score(line)
        elif lower.startswith("completeness"):
            scores["completeness"] = _extract_score(line)
        elif lower.startswith("usefulness"):
            scores["usefulness"] = _extract_score(line)
        elif lower.startswith("notes:"):
            scores["notes"] = line.split(":", 1)[1].strip()

    scores["raw"] = text
    return scores


def _extract_score(line: str) -> int | None:
    """Pull the trailing integer score off a line like 'Accuracy (1-5): 4'."""
    import re
    match = re.search(r"(\d+)\s*$", line)
    return int(match.group(1)) if match else None


def judge_all(product_batches: list[dict], llm_summaries: dict[str, str],
              config: LLMConfig | None = None) -> list[dict]:
    """
    Run judge_summary over a list of product batches.

    llm_summaries: dict mapping product_id -> summary string
    Returns a list of per-product result dicts.
    """
    config = config or LLMConfig()
    results = []
    for batch in product_batches:
        product_id = batch.get("product_id") or batch.get("asin") or batch.get("id")
        summary = llm_summaries.get(product_id, "")

        if not summary:
            results.append({"product_id": product_id, "error": "missing summary"})
            continue

        judgment = judge_summary(batch, summary, config=config)
        results.append({"product_id": product_id, "summary": summary, **judgment})
    return results
