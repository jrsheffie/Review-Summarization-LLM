# Evaluation: BART Baseline vs. Prompted LLM (Claude)

## Method
We compared two summarization approaches on a sample of product review batches:
1. **BART baseline** (`facebook/bart-large-cnn`), extractive/abstractive fine-tuned summarization
2. **Prompted LLM** (Claude, `claude-sonnet-4-6`), zero-shot structured summarization

Each pair was scored with:
- **ROUGE-L** overlap between the two summaries (lexical similarity signal)
- **LLM-as-judge**: Claude compares both summaries against the original reviews and picks a winner

## Preliminary Results (n=3 products)

| Product ID | ROUGE-L Overlap | Judge Winner |
|---|---|---|
| 0006641040 | 0.1231 | B (LLM) |
| 2734888454 | 0.0915 | B (LLM) |
| 7310172001 | 0.1064 | B (LLM) |

## Observations
- Low ROUGE-L overlap (~0.09–0.12) is expected: BART tends to extract near-verbatim sentences from the source reviews, while the prompted LLM produces a more abstractive, restructured summary — so lexical overlap between the two summaries is naturally low even when both are "correct."
- In all 3 preliminary cases, the LLM judge preferred the prompted-LLM summary, citing better synthesis across multiple reviews, clearer structure (sentiment/pros/cons/recommendation), and better handling of conflicting opinions within a product's reviews. BART's summaries tended to reproduce 1–2 source sentences with limited synthesis across the full review set.
- This is a small preliminary sample (n=3) intended to validate the evaluation pipeline before scaling. A full evaluation run across a larger sample is planned for the next milestone.

## Next Steps
- Scale evaluation to a larger, randomly sampled set of products (e.g., n=30–50)
- Consider adding standard ROUGE-1/ROUGE-2 against BART's own reference summaries if available
- Track Claude API cost/latency at scale for feasibility analysis
