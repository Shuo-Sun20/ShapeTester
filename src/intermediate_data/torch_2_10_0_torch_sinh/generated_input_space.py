import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(4),
    'out': None
}

# Task 4: Define InputSpace class with all parameters affecting output shape
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.randn(4),
        torch.randn(1, 4),
        torch.randn(2, 2),
        torch.randn(4, 1),
        torch.randn(1, 1, 4),
        torch.randn(2, 2, 1),
        torch.randn(1, 1, 1, 1),
        torch.randn(0, 4),
        torch.randn(4, 0)
    ])