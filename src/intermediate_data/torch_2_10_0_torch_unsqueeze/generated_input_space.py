import torch
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, dim):
    return torch.unsqueeze(inputs, dim)

# Valid test case
valid_test_case = {
    "inputs": torch.randn(3, 4),
    "dim": 1
}

@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: list(range(-7, 7)))