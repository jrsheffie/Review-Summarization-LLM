"""
bart_model.py  [Phase 2 — scaffold]

Loads facebook/bart-large-cnn and wraps LoRA adapters onto it via PEFT.
Actual loading/training happens on Colab (needs GPU + internet to pull
model weights from Hugging Face, unavailable in this dev sandbox).

TODO (Phase 2, on Colab):
    pip install transformers peft datasets

    from transformers import BartForConditionalGeneration, BartTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    from models.config import BartConfig, LoRAConfig

    def load_bart_with_lora(bart_cfg: BartConfig, lora_cfg: LoRAConfig):
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
"""

from models.config import BartConfig, LoRAConfig  # noqa: F401


def load_bart_with_lora(bart_cfg: BartConfig, lora_cfg: LoRAConfig):
    raise NotImplementedError(
        "Run on Colab with transformers + peft installed. See module docstring "
        "for the exact implementation to paste in during Phase 2."
    )
