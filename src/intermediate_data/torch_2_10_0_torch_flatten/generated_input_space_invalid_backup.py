import torch
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': torch.randn(2, 3, 4),
    'start_dim': 1,
    'end_dim': 2
}

@dataclass
class InputSpace:
    start_dim: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1])
    end_dim: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1])