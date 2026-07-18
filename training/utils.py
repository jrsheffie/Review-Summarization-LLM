"""
utils.py  [Phase 2 — scaffold]

Shared helpers for the training scripts (checkpoint saving/loading, seeding,
logging setup). Populate as training scripts are built out.
"""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
