import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Valid test case
valid_test_case = {
    'inputs': [torch.randn(3, 4), torch.randint(-5, 5, (3, 4), dtype=torch.int32)],
    'out': None
}

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros(3, 4),
        torch.ones(3, 4),
        torch.empty(3, 4),
        torch.full((3, 4), 2.0)
    ])