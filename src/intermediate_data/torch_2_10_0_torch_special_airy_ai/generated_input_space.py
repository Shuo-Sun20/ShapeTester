import torch
from dataclasses import dataclass, field

valid_test_case = {"inputs": torch.randn(3, 4), "out": None}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.empty(3, 4),
        torch.empty(1, 4),
        torch.empty(3, 1),
        torch.empty(4),
        torch.empty(1, 1)
    ])