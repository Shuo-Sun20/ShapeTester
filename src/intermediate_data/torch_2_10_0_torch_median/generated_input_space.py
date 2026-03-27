import torch
from dataclasses import dataclass, field

# 1. Define a valid test case
valid_test_case = {
    "inputs": torch.randn(4, 5),
    "dim": 1,
    "keepdim": False
}

# 2. & 3. & 4. Define the InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding "inputs")
    dim: list = field(default_factory=lambda: [None, -2, -1, 0, 1])
    keepdim: list = field(default_factory=lambda: [True, False])