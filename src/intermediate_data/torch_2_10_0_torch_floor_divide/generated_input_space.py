import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Valid test case
valid_test_case = {
    'inputs': [torch.randn(3, 4) * 10, torch.randn(3, 4) + 0.5],
    'out': None
}

# 2 & 3 & 4. InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(3, 4),
        torch.empty(1, 4),
        torch.empty(3, 1),
        torch.empty(1, 1)
    ])