import keras
import numpy as np
from dataclasses import dataclass

# Task 1: Define valid_test_case
batch_size = 2
Tq = 5
Tv = 7
dim = 3

query = np.random.randn(batch_size, Tq, dim).astype(np.float32)
value = np.random.randn(batch_size, Tv, dim).astype(np.float32)
key = np.random.randn(batch_size, Tv, dim).astype(np.float32)

valid_test_case = {
    "use_scale": True,
    "score_mode": "dot",
    "dropout": 0.1,
    "seed": 42,
    "inputs": [query, value, key],
    "mask": None,
    "return_attention_scores": False,
    "training": False,
    "use_causal_mask": False
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # No parameters in call_func (excluding inputs) affect the output tensor shape
    # The output shape is determined solely by input tensor dimensions
    pass