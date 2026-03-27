import torch
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
example_tensor = torch.randn(4, 4)
valid_test_case = {
    'inputs': example_tensor,
    'dim': 1,
    'descending': False,
    'stable': True
}

def call_func(inputs, dim=-1, descending=False, stable=False):
    return torch.argsort(inputs, dim=dim, descending=descending, stable=stable)

# Task 2: Parameters affecting output shape (except inputs): dim
# Task 3: Value spaces
# dim: discrete integer, typical values for a 4x4 tensor (2D)
# For a 4x4 tensor, valid dim values are -2, -1, 0, 1

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [-2, -1, 0, 1])