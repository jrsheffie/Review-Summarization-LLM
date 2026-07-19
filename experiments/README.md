# Experiments

Scratch space for exploration that doesn't belong in the official pipeline
notebooks (`notebooks/00_colab_setup.ipynb`, `notebooks/01_train_bart.ipynb`)
or the `training/` / `evaluation/` source modules.

## What goes here

- Trying different LoRA hyperparameters (rank, alpha, target modules) or
  training subset sizes before settling on the ones used in the official run
- Ad hoc timing/throughput checks
- One-off debugging (inspecting tokenization edge cases, checking a specific
  row, printing intermediate shapes)
- Exploratory comparisons that inform a decision documented elsewhere (e.g.
  a note in `docs/methodology.md`) but aren't themselves part of the graded
  pipeline

## Ground rules

- Notebooks here are **not** expected to run cleanly top-to-bottom or stay
  in sync with the rest of the repo -- that's the point of keeping them
  separate.
- If something explored here turns into a real decision (e.g. "we tried
  rank=8 vs rank=16 and picked 16 because..."), summarize the finding in the
  relevant `docs/` file rather than pointing reviewers back to a scratch
  notebook.
- Large outputs (checkpoints, generated CSVs) should stay out of git the
  same way they do elsewhere in this repo -- keep them in `outputs/` or your
  Drive backup, not committed from here.
