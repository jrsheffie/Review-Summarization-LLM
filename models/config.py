"""
config.py

Shared configuration for both the prompted LLM and fine-tuned BART model.
Centralized here so hyperparameters aren't scattered across scripts.
"""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 500
    temperature: float = 0.3  # lower = more consistent structured summaries


@dataclass
class BartConfig:
    base_model: str = "facebook/bart-large-cnn"
    max_input_length: int = 1024
    max_target_length: int = 128
    learning_rate: float = 2e-4
    batch_size: int = 8
    num_epochs: int = 3


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj")  # attention projections to adapt
