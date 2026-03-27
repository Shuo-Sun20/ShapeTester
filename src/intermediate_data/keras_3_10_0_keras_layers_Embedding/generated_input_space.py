import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
valid_test_case = {
    "input_dim": 1000,
    "output_dim": 64,
    "embeddings_initializer": "uniform",
    "embeddings_regularizer": None,
    "embeddings_constraint": None,
    "mask_zero": False,
    "weights": None,
    "lora_rank": None,
    "lora_alpha": None,
    "inputs": np.random.randint(low=0, high=1000, size=(32, 10))
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that affect output tensor shape (except inputs)
    output_dim: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])