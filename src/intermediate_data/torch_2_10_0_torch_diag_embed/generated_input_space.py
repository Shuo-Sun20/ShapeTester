import torch
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(2, 3)],
    "offset": 0,
    "dim1": -2,
    "dim2": -1
}

# Task 2, 3, & 4: Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    offset: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    dim1: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1])
    dim2: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1])