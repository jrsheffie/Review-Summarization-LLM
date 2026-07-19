"""
train_bart.py

Fine-tunes BART + LoRA on data/processed/train.csv and val.csv (produced by
data/preprocess.py). Input column: Text (review body). Target column:
Summary (per-review headline). Intended to run on Colab with a GPU runtime.

Saves the LoRA adapter (not the full model) to outputs/bart_lora_checkpoint/,
which is small enough to keep out of git history but cheap to reload later
via PeftModel.from_pretrained on top of the base facebook/bart-large-cnn.
"""

from __future__ import annotations

import argparse

import pandas as pd
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from models.bart_model import load_bart_with_lora
from models.config import BartConfig, LoRAConfig


def load_split(path: str) -> pd.DataFrame:
    """Load a train/val/test CSV, tolerating the occasional malformed row
    (e.g. a truncated final line) rather than crashing outright."""
    df = pd.read_csv(path, engine="python", on_bad_lines="warn")
    df = df.dropna(subset=["Text", "Summary"])
    df["Text"] = df["Text"].astype(str)
    df["Summary"] = df["Summary"].astype(str)
    return df


def build_tokenize_fn(tokenizer, bart_cfg: BartConfig):
    def tokenize(batch):
        model_inputs = tokenizer(
            batch["Text"],
            max_length=bart_cfg.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["Summary"],
            max_length=bart_cfg.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BART + LoRA on review-summary pairs.")
    parser.add_argument("--train-path", default="data/processed/train.csv", help="Training CSV path")
    parser.add_argument("--val-path", default="data/processed/val.csv", help="Validation CSV path")
    parser.add_argument("--output-dir", default="outputs/bart_lora_checkpoint", help="Checkpoint output directory")
    return parser.parse_args()


def train(train_path: str = "data/processed/train.csv", val_path: str = "data/processed/val.csv",
           output_dir: str = "outputs/bart_lora_checkpoint"):
    bart_cfg = BartConfig()
    lora_cfg = LoRAConfig()

    print("Loading BART + LoRA...")
    model, tokenizer = load_bart_with_lora(bart_cfg, lora_cfg)

    print("Loading train/val splits...")
    train_df = load_split(train_path)
    val_df = load_split(val_path)
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    train_ds = Dataset.from_pandas(train_df[["Text", "Summary"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["Text", "Summary"]], preserve_index=False)

    tokenize_fn = build_tokenize_fn(tokenizer, bart_cfg)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["Text", "Summary"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["Text", "Summary"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=bart_cfg.batch_size,
        per_device_eval_batch_size=bart_cfg.batch_size,
        learning_rate=bart_cfg.learning_rate,
        num_train_epochs=bart_cfg.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,  # Colab T4 supports fp16; speeds up training substantially
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    final_dir = f"{output_dir}/final"
    print("Saving final LoRA adapter...")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Done. Adapter saved to {final_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(train_path=args.train_path, val_path=args.val_path, output_dir=args.output_dir)
