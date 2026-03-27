import torch
from dataclasses import dataclass, field
from typing import Optional

valid_test_case = {
    'inputs': torch.randn(3, 3),
    'out': None
}

@dataclass
class InputSpace:
    out: list[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(3, 3),
        torch.zeros(3, 3),
        torch.ones(3, 3),
        torch.randn(3, 3)
    ])