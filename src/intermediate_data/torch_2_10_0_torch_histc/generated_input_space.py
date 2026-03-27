import torch
from dataclasses import dataclass, field

valid_test_case = {
    'inputs': torch.randn(50),
    'bins': 20,
    'min': -2.0,
    'max': 2.0,
    'out': None
}

@dataclass
class InputSpace:
    bins: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20])