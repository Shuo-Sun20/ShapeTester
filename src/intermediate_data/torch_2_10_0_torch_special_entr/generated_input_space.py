import torch
from dataclasses import dataclass, field

# 1. Define a valid test case
valid_test_case = {
    "inputs": torch.randn(3, 4),
    "out": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [None] + [
        torch.empty(3, 4),
        torch.empty(5, 2),
        torch.empty(2, 2, 2),
        torch.empty(1),
        torch.empty(0, 3)
    ])