import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.rand(5),
    "eps": 1e-6,
    "out": None
}

# 2. Parameters affecting output shape (excluding inputs): out

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.empty(5),
            torch.empty(1, 5),
            torch.empty(5, 1),
            torch.empty(1, 1, 5)
        ]
    )