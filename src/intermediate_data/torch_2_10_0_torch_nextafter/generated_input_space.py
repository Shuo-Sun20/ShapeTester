import torch
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": [torch.randn(3, 4), torch.randn(3, 4)],
    "out": None
}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.zeros(3, 4),
        torch.zeros(3, 4, dtype=torch.float16),
        torch.empty(3, 4),
        torch.full((3, 4), fill_value=0.5)
    ])