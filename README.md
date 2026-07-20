# Review-Summarization-LLM

**Summarizing Customer Product Reviews Using Generative AI**
IE7374 Final Project — Josiah Sheffie

## Overview

This project compares two generative approaches to abstractive summarization
of Amazon product reviews:

1. **Zero-shot prompted LLM** (Claude) — a decoder-only autoregressive
   transformer, used purely through prompt engineering with no task-specific
   training.
2. **Fine-tuned BART + LoRA** — an encoder-decoder model pretrained with a
   denoising objective, fine-tuned on review-summary pairs using
   parameter-efficient LoRA adapters.

The comparison highlights the tradeoff between a general-purpose zero-shot
model and a lightly fine-tuned, task-specific one. See `docs/methodology.md`
for full details, `docs/literature_review.md` for supporting research, and
`docs/model_comparison.md` for the full benchmarking analysis.

## Repository Structure

    .
    ├── data/
    │   ├── raw/                    # Raw dataset (gitignored; sample included)
    │   ├── processed/               # Cleaned/split data (generated)
    │   ├── download_dataset.py      # Kaggle download script
    │   ├── preprocess.py            # Cleaning, filtering, batching, splitting - tested
    │   └── exploratory_analysis.py  # EDA: distributions, missing data, figures - tested
    ├── models/
    │   ├── config.py                # Shared hyperparameter configs
    │   ├── llm_prompting.py         # Claude API summarization pipeline - built
    │   └── bart_model.py            # BART + LoRA loading via PEFT - built
    ├── training/
    │   ├── train_bart.py            # BART + LoRA fine-tuning loop - built
    │   ├── inference.py             # Beam-search generation (Phase 2)
    │   └── utils.py                 # Seeding, shared helpers
    ├── evaluation/
    │   ├── evaluate.py              # Top-level evaluation runner (reference + llm-judge modes) - built
    │   ├── rouge_metrics.py         # ROUGE-1/2/L against reference summaries - built
    │   ├── bertscore_metrics.py     # BERTScore against reference summaries - built
    │   ├── llm_judge.py             # Claude-as-judge for prompted-LLM summaries (no reference) - built
    │   └── manual_evaluation.py     # Human rubric scoring scaffold
    ├── notebooks/
    │   ├── 00_colab_setup.ipynb     # Run first each Colab session - ready
    │   ├── 01_train_bart.ipynb      # Official BART + LoRA training run (30k-row subset) - ready
    │   ├── 02_evaluate_bart.ipynb   # Official BART evaluation (ROUGE + BERTScore) - ready
    │   └── 03_manual_eval.ipynb     # Manual rubric cross-model comparison - ready
    ├── experiments/
    │   └── README.md                # Scratch space for hyperparameter exploration, ad hoc checks
    ├── docs/
    │   ├── methodology.md           # Approach, task-granularity decision, evaluation design
    │   ├── literature_review.md     # Verified summaries of 7 supporting papers - complete
    │   ├── model_comparison.md      # Benchmarking: accuracy, cost, scalability, reproducibility
    │   ├── evaluation_report.md     # Full evaluation: automatic metrics + manual rubric - complete
    │   ├── pipeline_documentation.md
    │   └── timeline.md
    ├── outputs/
    │   ├── summaries/, figures/, metrics/   # Generated artifacts (gitignored)
    │   └── bart_lora_*/                     # Training checkpoints (gitignored; backed up to Drive)
    ├── tests/
    │   └── test_preprocess.py       # 8 passing unit tests
    ├── requirements.txt
    └── .gitignore

**Status legend:** built & tested this milestone (marked "built"/"tested"/"complete") · scaffolded, implementation pending (marked "Phase 2"/"in progress") · not started. See `docs/timeline.md` for the full breakdown.

## Setup (Google Colab)

1. Open `notebooks/00_colab_setup.ipynb` in Colab
2. Runtime → Change runtime type → GPU (T4 is fine)
3. Run all cells: mounts Drive, clones this repo, installs `requirements.txt`,
   confirms GPU, sets up API keys, downloads the dataset
4. For fine-tuning BART, run `notebooks/01_train_bart.ipynb` afterward — it
   trains on a fixed 30k-row subset (~30-45 min on a T4) and backs the
   resulting checkpoint up to Drive
5. For evaluation, run `notebooks/02_evaluate_bart.ipynb` (automatic metrics)
   and `notebooks/03_manual_eval.ipynb` (manual rubric comparison)

## Setup (local, alternative)

    git clone https://github.com/jrsheffie/Review-Summarization-LLM.git
    cd Review-Summarization-LLM
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## API Keys

This project calls the Anthropic API for the prompted-LLM approach. Set your
key as an environment variable — never commit it to the repo:

    export ANTHROPIC_API_KEY="your-key-here"

In Colab, store it under Secrets (key icon) as `ANTHROPIC_API_KEY` and load it
with `google.colab.userdata.get('ANTHROPIC_API_KEY')` — see
`notebooks/00_colab_setup.ipynb`.

## Getting the Dataset

Dataset: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) (Kaggle)

    # One-time: place your kaggle.json in ~/.kaggle/ (see data/download_dataset.py docstring)
    pip install kaggle
    python data/download_dataset.py

A small synthetic sample (`data/raw/sample_reviews.csv`) is included so the
pipeline can be run immediately without the full dataset.

The dataset includes a `Summary` column (a short, human-written headline per
individual review), which serves as the ground-truth target for BART
fine-tuning and reference-based evaluation. The prompted LLM operates at a
different granularity (multi-review, product-level synthesis) and has no
direct reference summary — see `docs/methodology.md` for how this is handled
in training and evaluation.

## Running the Pipeline

    # Clean, filter, batch, and split the data
    python data/preprocess.py \
        --input data/raw/sample_reviews.csv \
        --output-dir data/processed

    # Exploratory analysis
    python data/exploratory_analysis.py \
        --input data/raw/sample_reviews.csv \
        --output-dir outputs/figures

## Running BART Fine-Tuning

    # Full dataset (defaults to data/processed/train.csv and val.csv)
    python -m training.train_bart

    # Custom subset (used by notebooks/01_train_bart.ipynb for the official run)
    python -m training.train_bart \
        --train-path data/processed/train_subset_30k.csv \
        --val-path data/processed/val_subset_30k.csv \
        --output-dir outputs/bart_lora_30k

Training uses the hyperparameters in `models/config.py`'s `BartConfig` and
`LoRAConfig`. For alternate subset sizes or hyperparameter exploration, see
`experiments/`.

## Running Evaluation

    # BART vs. reference Summary column (ROUGE + BERTScore)
    python -m evaluation.evaluate reference \
        --predictions outputs/summaries/bart_predictions.csv \
        --pred-col prediction --ref-col Summary \
        --output outputs/metrics/bart_results.json

    # Prompted LLM vs. original reviews (Claude-as-judge, no reference needed)
    python -m evaluation.evaluate llm-judge \
        --batches data/processed/product_batches.json \
        --summaries outputs/summaries/llm_summaries.json \
        --output outputs/metrics/llm_judge_results.json

See `docs/evaluation_report.md` for full results, including the manual
rubric cross-model comparison.

## Running Tests

    pytest tests/ -v

## Roadmap

- [x] **Milestone 3** — Repo structure, README, data pipeline (built & tested), EDA, documentation, literature review
- [x] **Phase 2** — Prompted-LLM API wired in; BART + LoRA fine-tuned on a 30k-row subset (real loss curve, see `01_train_bart.ipynb`)
- [x] **Phase 3** — Evaluation complete: ROUGE, BERTScore, LLM-judge, and manual rubric cross-model comparison (see `docs/evaluation_report.md`)
- [ ] **Phase 4** — Final report and presentation
