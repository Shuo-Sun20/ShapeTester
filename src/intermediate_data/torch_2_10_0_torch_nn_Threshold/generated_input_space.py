import torch
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "threshold": 0.5,
    "value": 0.1,
    "inplace": False,
    "inputs": torch.randn(3, 4)
}

@dataclass
class InputSpace:
    threshold: List[float] = field(default_factory=lambda: [
        -float('inf'), -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, float('inf')
    ])
    value: List[float] = field(default_factory=lambda: [
        -float('inf'), -1.5, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 1.5, float('inf')
    ])
    inplace: List[bool] = field(default_factory=lambda: [True, False])