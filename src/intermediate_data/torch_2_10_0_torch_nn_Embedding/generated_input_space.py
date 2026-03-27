import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Valid test case dictionary
valid_test_case = {
    "num_embeddings": 100,
    "embedding_dim": 64,
    "padding_idx": None,
    "max_norm": None,
    "norm_type": 2.0,
    "scale_grad_by_freq": False,
    "sparse": False,
    "inputs": torch.randint(0, 100, (4, 10))
}

# 2. & 3. Parameters affecting output shape and their value spaces
# Only embedding_dim affects output shape directly
# Parameter types and value spaces:
# - embedding_dim: positive integer (continuous -> discretized)
# Note: Other parameters like num_embeddings don't affect output shape,
# only whether indices are valid

@dataclass
class InputSpace:
    embedding_dim: List[int] = field(default_factory=lambda: [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    ])