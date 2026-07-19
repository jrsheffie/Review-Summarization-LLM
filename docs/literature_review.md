# Literature Review

*Summaries below have been verified against the actual paper abstracts/sources
as of this milestone.*

## 1. BART (Lewis et al., 2019/2020 — "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension")
**Key contribution:** Introduces a sequence-to-sequence pretraining objective that corrupts text (token masking, deletion, sentence permutation, document rotation, text infilling) and trains an encoder-decoder transformer to reconstruct the original. Combines BERT's bidirectional encoder with GPT's autoregressive decoder, and matches RoBERTa's performance on GLUE/SQuAD while achieving state-of-the-art results on abstractive summarization tasks.
**Relevance to this project:** BART is the exact architecture being fine-tuned in this project's second approach; its denoising pretraining is why it transfers well to abstractive summarization specifically, motivating its selection over encoder-only or decoder-only alternatives.
**Citation:** Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, 7871–7880.

## 2. T5 (Raffel et al., 2020 — "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer")
**Key contribution:** Reframes every NLP task — classification, translation, summarization — as a text-to-text problem, and systematically studies pretraining objectives and model/data scale on this unified framework.
**Relevance to this project:** Serves as a conceptual point of contrast to BART in this project's methodology — both are encoder-decoder models suited to summarization, but T5's text-to-text framing differs from BART's denoising-reconstruction framing, which is worth noting when justifying the choice of BART specifically.
**Citation:** Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *Journal of Machine Learning Research*, 21(140), 1-67.

## 3. LoRA (Hu et al., 2021 — "LoRA: Low-Rank Adaptation of Large Language Models")
**Key contribution:** Freezes the pretrained model's weights and injects small trainable low-rank decomposition matrices into each layer of the Transformer, drastically reducing the number of trainable parameters needed for fine-tuning (up to 10,000x fewer than full fine-tuning of GPT-3 175B, with 3x less GPU memory) while matching or exceeding full fine-tuning performance on RoBERTa, DeBERTa, GPT-2, and GPT-3.
**Relevance to this project:** Directly enables this project's fine-tuning approach — training BART on review-summary pairs would be far less feasible on a single Colab GPU with full fine-tuning; LoRA is what makes the comparison to a zero-shot LLM practical to run at all.
**Citation:** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.

## 4. Opinion Mining / Review Summarization (Hu & Liu, 2004 — "Mining and Summarizing Customer Reviews")
**Key contribution:** One of the foundational, test-of-time-award-winning papers in aspect-based opinion mining — decomposes review summarization into three steps: mining product features (aspects), identifying opinion sentences and their corresponding feature in each review, and summarizing the results, rather than summarizing reviews as generic text.
**Relevance to this project:** Establishes the domain-specific precedent for structured review summarization (e.g. pros/cons by feature) that this project's prompt template design (sentiment, pros, cons, recommendation) draws on, even though this project uses generative rather than extractive/aspect-mining methods.
**Citation:** Hu, M., & Liu, B. (2004). Mining and Summarizing Customer Reviews. *Proceedings of the 10th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 168-177.

## 5. Zero-Shot LLM Summarization (Goyal, Li, & Durrett, 2022 — "News Summarization and Evaluation in the Era of GPT-3")
**Key contribution:** Shows that humans overwhelmingly prefer GPT-3 summaries produced from only a task-description prompt over summaries from models fine-tuned on large summarization datasets, and that these zero-shot summaries avoid common dataset-specific issues such as poor factuality. Critically, it also shows that standard automatic evaluation metrics (reference-based and reference-free) fail to reliably evaluate GPT-3-quality summaries.
**Relevance to this project:** Directly motivates this project's first approach (zero-shot prompted summarization) and its core research question — how does zero-shot prompting compare to lightweight fine-tuning for this task? It also directly justifies this project's evaluation design choice to supplement ROUGE/BERTScore with a manual rubric and LLM-judge (see `docs/methodology.md`), since this paper specifically found automatic metrics inadequate for judging prompted-LLM output.
**Citation:** Goyal, T., Li, J. J., & Durrett, G. (2022). News Summarization and Evaluation in the Era of GPT-3. *arXiv:2209.12356*.

## 6. ROUGE (Lin, 2004 — "ROUGE: A Package for Automatic Evaluation of Summaries")
**Key contribution:** Introduces n-gram and longest-common-subsequence overlap metrics (ROUGE-N, ROUGE-L) for automatically evaluating generated summaries against reference summaries, correlating reasonably well with human judgment at the time.
**Relevance to this project:** One of the automated metrics used in this project's evaluation of BART against real reference summaries (the dataset's `Summary` column); its lexical-overlap nature is also why the project supplements it with BERTScore and a manual rubric (see `docs/methodology.md` and `evaluation/manual_evaluation.py`) — consistent with Goyal et al. (2022)'s finding that automatic metrics alone are insufficient, especially for the prompted-LLM side of this comparison.
**Citation:** Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out*, 74-81.

## 7. BERTScore (Zhang et al., 2020 — "BERTScore: Evaluating Text Generation with BERT")
**Key contribution:** Proposes computing similarity between generated and reference text using contextual token embeddings from a pretrained BERT-family model rather than surface-level n-gram overlap, and shows it correlates better with human judgments than existing metrics across machine translation and image captioning systems, while being more robust on an adversarial paraphrase detection task.
**Relevance to this project:** Used alongside ROUGE precisely because it captures cases where a generated summary is semantically correct but lexically different from the reference (e.g. paraphrased pros/cons) — a known weakness of ROUGE alone, and particularly relevant given the abstractive, restructured nature of both models' outputs in this project.
**Citation:** Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)*.
