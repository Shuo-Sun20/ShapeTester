import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Valid test case
valid_test_case = {
    "inputs": torch.randn(2, 3),
    "dim": 1,
    "keepdim": False,
    "dtype": None
}

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    dim: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 3])
    keepdim: List[bool] = field(default_factory=lambda: [True, False])