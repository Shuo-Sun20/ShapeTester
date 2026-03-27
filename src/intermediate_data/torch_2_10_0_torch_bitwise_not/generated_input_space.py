import torch
from dataclasses import dataclass, field
from typing import Optional

valid_test_case = {
    "inputs": torch.randint(-10, 10, (5,), dtype=torch.int8),
    "out": None
}

@dataclass
class InputSpace:
    out: Optional[list] = field(default_factory=lambda: [
        None,
        torch.empty((5,), dtype=torch.int8),
        torch.empty((1, 5), dtype=torch.int8),
        torch.empty((5, 1), dtype=torch.int8),
        torch.empty((1, 1, 5), dtype=torch.int8)
    ])