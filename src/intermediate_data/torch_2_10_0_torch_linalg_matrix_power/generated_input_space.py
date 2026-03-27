import torch
from dataclasses import dataclass, field
from typing import Optional

def call_func(inputs, n, out=None):
    return torch.linalg.matrix_power(inputs, n, out=out)

valid_test_case = {
    'inputs': torch.randn(3, 3),
    'n': 3,
    'out': None
}

@dataclass
class InputSpace:
    n: list[int] = field(default_factory=lambda: [-5, -3, -2, -1, 0, 1, 2, 3, 5])
    out: list[Optional[torch.Tensor]] = field(default_factory=lambda: [None])