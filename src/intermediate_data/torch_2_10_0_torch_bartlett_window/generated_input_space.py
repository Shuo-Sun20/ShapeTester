import torch
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "window_length": 10,
    "periodic": True,
    "inputs": None,
    "dtype": torch.float32,
    "layout": torch.strided,
    "device": "cpu",
    "requires_grad": False
}

@dataclass
class InputSpace:
    window_length: List[int] = field(default_factory=lambda: [1, 2, 10, 50, 100])
    periodic: List[bool] = field(default_factory=lambda: [True, False])