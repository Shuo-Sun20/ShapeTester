import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case
mat = torch.randn(2, 3)
vec = torch.randn(3)
valid_test_case = {
    'inputs': [mat, vec],
    'out': None
}

# 4. Define InputSpace class
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0])
    ])