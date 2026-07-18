"""
train_bart.py  [Phase 2 — scaffold]

Fine-tunes BART + LoRA on data/processed/train.csv and val.csv (produced by
data/preprocess.py). Intended to run on Colab with a GPU runtime.

TODO (Phase 2):
    - Load train.csv / val.csv into a Hugging Face Dataset
    - Tokenize reviews (input) and summaries (target) -- NOTE: this requires
      ground-truth summaries. If the raw dataset has no reference summaries,
      the LLM's zero-shot outputs (or review titles, as a proxy) may need to
      serve as weak-supervision targets -- decide this before Phase 2 and
      document the choice in docs/methodology.md.
    - Use transformers.Seq2SeqTrainer with the LoRA-wrapped model from
      models/bart_model.py
    - Save checkpoints to outputs/ (gitignored -- too large for GitHub)
"""

from models.bart_model import load_bart_with_lora
from models.config import BartConfig, LoRAConfig


def train():
    bart_cfg = BartConfig()
    lora_cfg = LoRAConfig()
    model, tokenizer = load_bart_with_lora(bart_cfg, lora_cfg)
    raise NotImplementedError("Training loop to be implemented in Phase 2 on Colab.")


if __name__ == "__main__":
    train()
