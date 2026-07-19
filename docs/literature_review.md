# Literature Review

*Draft based on well-established, widely-cited contributions of each paper.
Verify each summary against the actual paper before final submission —
these are drafted from general knowledge of the field, not from re-reading
the papers directly alongside you, so page-specific details or exact
figures should be double-checked.*

## 1. BART (Lewis et al., 2019 — "BART: Denoising Sequence-to-Sequence Pre-training...")
**Key contribution:** Introduces a sequence-to-sequence pretraining objective that corrupts text (token masking, deletion, sentence permutation, document rotation) and trains an encoder-decoder transformer to reconstruct the original. Combines BERT's bidirectional encoder with GPT's autoregressive decoder.
**Relevance to this project:** BART is the exact architecture being fine-tuned in this project's second approach; its denoising pretraining is why it transfers well to abstractive summarization specifically, motivating its selection over encoder-only or decoder-only alternatives.

## 2. T5 (Raffel et al., 2020 — "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer")
**Key contribution:** Reframes every NLP task — classification, translation, summarization — as a text-to-text problem, and systematically studies pretraining objectives and model/data scale on this unified framework.
**Relevance to this project:** Serves as a conceptual point of contrast to BART in this project's methodology — both are encoder-decoder models suited to summarization, but T5's text-to-text framing differs from BART's denoising-reconstruction framing, which is worth noting when justifying the choice of BART specifically.

## 3. LoRA (Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models")
**Key contribution:** Freezes the pretrained model's weights and injects small trainable low-rank matrices into attention layers, drastically reducing the number of trainable parameters needed for fine-tuning while matching full fine-tuning performance in many settings.
**Relevance to this project:** Directly enables this project's fine-tuning approach — training BART on review-summary pairs would be far less feasible on a single Colab GPU with full fine-tuning; LoRA is what makes the comparison to a zero-shot LLM practical to run at all.

## 4. Opinion Mining / Review Summarization (Hu & Liu, 2004 — "Mining and Summarizing Customer Reviews")
**Key contribution:** One of the foundational papers in aspect-based opinion mining — extracting product features mentioned in reviews and the sentiment expressed toward each, rather than summarizing reviews as generic text.
**Relevance to this project:** Establishes the domain-specific precedent for structured review summarization (e.g. pros/cons by feature) that this project's prompt template design (sentiment, pros, cons, recommendation) draws on, even though this project uses generative rather than extractive methods.

## 5. LLM-based Summarization (general — e.g. GPT-3/GPT-4 technical reports, or survey papers on zero-shot summarization)
**Key contribution:** Demonstrates that large pretrained decoder-only LLMs can perform competitive abstractive summarization purely through prompting, without task-specific fine-tuning, given sufficiently descriptive instructions.
**Relevance to this project:** Directly motivates this project's first approach (zero-shot prompted summarization) and the framing of the core research question — how does zero-shot prompting compare to lightweight fine-tuning for this task?
*(Recommend picking one specific paper here rather than leaving this general — e.g. a specific zero-shot summarization evaluation paper — since "general" citations are weaker in a literature review.)*

## 6. ROUGE (Lin, 2004 — "ROUGE: A Package for Automatic Evaluation of Summaries")
**Key contribution:** Introduces n-gram and longest-common-subsequence overlap metrics (ROUGE-N, ROUGE-L) for automatically evaluating generated summaries against reference summaries, correlating reasonably well with human judgment at the time.
**Relevance to this project:** One of the two automated metrics used in this project's evaluation; its lexical-overlap nature is also why the project supplements it with BERTScore and a manual rubric (see `docs/methodology.md` and `evaluation/manual_evaluation.py`).

## 7. BERTScore (Zhang et al., 2020 — "BERTScore: Evaluating Text Generation with BERT")
**Key contribution:** Proposes computing similarity between generated and reference text using contextual token embeddings from a pretrained BERT-family model, rather than surface-level n-gram overlap, better capturing paraphrases and semantic equivalence.
**Relevance to this project:** Used alongside ROUGE precisely because it captures cases where a generated summary is semantically correct but lexically different from the reference (e.g. paraphrased pros/cons) — a known weakness of ROUGE alone.

---

**To finish this section:** pull up each paper's actual abstract (all are freely available — BART, T5, and LoRA are on arXiv; ROUGE and BERTScore are widely available via ACL Anthology / arXiv) and adjust any details above that don't match, then add the full citation (authors, venue, year) in whatever citation style your course requires.
