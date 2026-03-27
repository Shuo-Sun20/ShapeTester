import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        torch.tensor([-1, -2, 3], dtype=torch.int8),
        torch.tensor([1, 0, 3], dtype=torch.int8)
    ],
    "out": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros((3, 4), dtype=torch.int8),
        torch.ones((3, 4), dtype=torch.int8),
        torch.full((3, 4), -1, dtype=torch.int8),
        torch.empty((3, 4), dtype=torch.int8)
    ])