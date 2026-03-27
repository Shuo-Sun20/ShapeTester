import torch
from dataclasses import dataclass, field
from typing import Optional

valid_test_case = {
    'inputs': torch.randn(3, 4),
    'out': None
}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.empty(0),
        torch.empty(1),
        torch.empty(3, 4),
        torch.empty(2, 3, 5),
        torch.empty(1, 2, 3, 4),
        torch.tensor(5.0),
        torch.full((2, 3), float('inf')),
        torch.full((2, 3), float('-inf')),
        torch.full((2, 3), float('nan'))
    ])