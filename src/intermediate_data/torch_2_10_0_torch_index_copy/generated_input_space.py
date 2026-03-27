from dataclasses import dataclass, field
from typing import Optional, List
import torch

valid_test_case = {
    'inputs': [torch.randn(5, 3), torch.randint(0, 5, (3,)), torch.randn(3, 3)],
    'dim': 0,
    'out': None
}

@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [None] + [
            torch.empty(5, 3),
            torch.empty(5, 3),
            torch.empty(5, 3),
            torch.empty(5, 3)
        ]
    )