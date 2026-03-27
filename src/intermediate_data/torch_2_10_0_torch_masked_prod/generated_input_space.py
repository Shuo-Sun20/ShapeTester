import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

valid_test_case = {
    "inputs": [torch.randn(3, 4)],
    "dim": 1,
    "keepdim": False,
    "dtype": None,
    "mask": torch.randint(0, 2, (3, 4), dtype=torch.bool)
}

@dataclass
class InputSpace:
    dim: list = field(default_factory=lambda: [None, 0, 1, -1, -2, (0, 1), (0,), (1,), (-1,), (-2,), (-1, -2)])
    keepdim: list = field(default_factory=lambda: [True, False])