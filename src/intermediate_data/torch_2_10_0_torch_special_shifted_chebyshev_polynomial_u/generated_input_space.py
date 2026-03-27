import torch
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": [torch.randn(3, 3), torch.tensor(3)],
    "out": None
}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.empty(3, 3),
        torch.empty(2, 4),
        torch.empty(5),
        torch.empty(1, 1, 1),
        torch.empty(0)
    ])