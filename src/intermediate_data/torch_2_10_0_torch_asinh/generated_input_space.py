import torch
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    "inputs": [torch.randn(4)],
    "out": None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(4),
        torch.empty(2, 3),
        torch.empty(1, 1, 1),
        torch.empty(0, 2, 3)
    ])