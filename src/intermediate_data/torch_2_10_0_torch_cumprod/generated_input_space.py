import torch
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    'inputs': [torch.randn(10)],
    'dim': 0,
    'dtype': None,
    'out': None
}

@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])