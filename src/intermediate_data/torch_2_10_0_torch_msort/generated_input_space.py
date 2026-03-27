import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(3, 4)],
    'out': None
}

# 4. Define InputSpace class
@dataclass
class InputSpace:
    # Only parameter affecting output shape (besides inputs) is 'out'
    # Value space includes None and 4 tensor configurations with correct shape
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.zeros(3, 4),
            torch.ones(3, 4),
            torch.randn(3, 4),
            torch.full((3, 4), 2.5)
        ]
    )