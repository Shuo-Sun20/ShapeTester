import torch
from dataclasses import dataclass, field
from typing import Union

valid_test_case = {
    "inputs": torch.randn(4),
    "out": None
}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.empty(1),
        torch.empty(2),
        torch.empty(3),
        torch.empty(4),
        torch.empty(5)
    ])