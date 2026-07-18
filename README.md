# Review-Summarization-LLM

**Summarizing Customer Product Reviews Using Generative AI**
IE7374 Final Project — Josiah Sheffie

## Overview

This project compares two generative approaches to abstractive summarization
of Amazon product reviews:

1. **Zero-shot prompted LLM** (Claude or GPT-4) — a decoder-only autoregressive
   transformer, used purely through prompt engineering with no task-specific
   training.
2. **Fine-tuned BART + LoRA** — an encoder-decoder model pretrained with a
   denoising objective, fine-tuned on review-summary pairs using
   parameter-efficient LoRA adapters.

The comparison highlights the tradeoff between a general-purpose zero-shot
model and a lightly fine-tuned, task-specific one. See `docs/methodology.md`
for full details.

## Repository Structure

```
.
├── data/
│   ├── raw/                    # Raw dataset (gitignored; sample included)
│   ├── processed/               # Cleaned/split data (generated)
│   ├── download_dataset.py      # Kaggle download script
│   ├── preprocess.py            # Cleaning, filtering, batching, splitting ✅ tested
│   └── exploratory_analysis.py  # EDA: distributions, missing data, figures ✅ tested
├── models/
│   ├── config.py                # Shared hyperparameter configs
│   ├── llm_prompting.py         # Prompt template + batching (Phase 2)
│   └── bart_model.py            # BART + LoRA loading (Phase 2, needs Colab GPU)
├── training/
│   ├── train_bart.py            # Fine-tuning loop (Phase 2)
│   ├── inference.py             # Beam-search generation (Phase 2)
│   └── utils.py                 # Seeding, shared helpers
├── evaluation/
│   ├── evaluate.py              # Top-level evaluation runner (Phase 3)
│   ├── rouge_metrics.py         # ROUGE-1/2/L (Phase 3)
│   ├── bertscore_metrics.py     # BERTScore (Phase 3)
│   └── manual_evaluation.py     # Human rubric scoring (Phase 3)
├── notebooks/
│   └── 00_colab_setup.ipynb     # Run first each Colab session ✅ ready
├── docs/
│   ├── methodology.md
│   ├── timeline.md
│   ├── literature_review.md     # Template — fill in per-paper summaries
│   └── pipeline_documentation.md
├── outputs/
│   ├── summaries/, figures/, metrics/   # Generated artifacts (gitignored)
├── tests/
│   └── test_preprocess.py       # 8 passing unit tests
├── requirements.txt
└── .gitignore
```

**Status legend:** ✅ built & tested this milestone · 🟡 scaffolded, implementation pending · ⬜ not started. See `docs/timeline.md` for the full breakdown.

## Setup (Google Colab)

1. Open `notebooks/00_colab_setup.ipynb` in Colab
2. Runtime → Change runtime type → GPU (T4 is fine)
3. Run all cells: mounts Drive, clones this repo, installs `requirements.txt`,
   confirms GPU, sets up Kaggle API, downloads the dataset

## Setup (local, alternative)

```bash
git clone https://github.com/<your-username>/Review-Summarization-LLM.git
cd Review-Summarization-LLM
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Getting the Dataset

Dataset: [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) (Kaggle)

```bash
# One-time: place your kaggle.json in ~/.kaggle/ (see data/download_dataset.py docstring)
pip install kaggle
python data/download_dataset.py
```

A small synthetic sample (`data/raw/sample_reviews.csv`) is included so the
pipeline can be run immediately without the full dataset.

**⚠️ Open item:** the Kaggle dataset has no reference/gold summaries needed
for BART fine-tuning and ROUGE evaluation. See "Open Question" in
`docs/methodology.md` for the options being considered — resolve this before
starting Phase 2.

## Running the Pipeline

```bash
# Clean, filter, batch, and split the data
python data/preprocess.py \
    --input data/raw/sample_reviews.csv \
    --output-dir data/processed

# Exploratory analysis
python data/exploratory_analysis.py \
    --input data/raw/sample_reviews.csv \
    --output-dir outputs/figures
```

## Running Tests

```bash
pytest tests/ -v
```

## Roadmap

- [x] **Milestone 3** — Repo structure, README, data pipeline (built & tested), EDA, documentation
- [ ] **Phase 2** — Wire in LLM API calls, fine-tune BART + LoRA on Colab
- [ ] **Phase 3** — ROUGE/BERTScore evaluation, manual rubric scoring
- [ ] **Phase 4** — Final report and presentation
