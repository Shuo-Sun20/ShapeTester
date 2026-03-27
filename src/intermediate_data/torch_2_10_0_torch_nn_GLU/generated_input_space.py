import torch
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': torch.randn(4, 2),
    'dim': -1
}

@dataclass
class InputSpace:
    dim: List[int] = field(
        default_factory=lambda: [-4, -3, -2, -1, 0, 1, 2, 3]
    )