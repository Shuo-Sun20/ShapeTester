import torch
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "inputs": [torch.randn(3), torch.randn(3)],
    "rtol": 1e-05,
    "atol": 1e-08,
    "equal_nan": False
}

@dataclass
class InputSpace:
    rtol: List[float] = field(default_factory=lambda: [0.0, 1e-10, 1e-5, 0.1, 1.0])
    atol: List[float] = field(default_factory=lambda: [0.0, 1e-12, 1e-8, 0.001, 1.0])
    equal_nan: List[bool] = field(default_factory=lambda: [False, True])