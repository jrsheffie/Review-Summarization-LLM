"""
bart_model.py

Loads facebook/bart-large-cnn and wraps LoRA adapters onto it via PEFT.
"""

from transformers import BartForConditionalGeneration, BartTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from models.config import BartConfig, LoRAConfig


def load_bart_with_lora(bart_cfg: BartConfig, lora_cfg: LoRAConfig):
    """Load the base BART model + tokenizer and wrap it with LoRA adapters."""
    tokenizer = BartTokenizer.from_pretrained(bart_cfg.base_model)
    model = BartForConditionalGeneration.from_pretrained(bart_cfg.base_model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=list(lora_cfg.target_modules),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer
