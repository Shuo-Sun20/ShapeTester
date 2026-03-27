import torch
from dataclasses import dataclass, field
from typing import Optional, List
from torch import Generator

valid_test_case = {
    'inputs': torch.rand(3, 3),
    'generator': None,
    'out': None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(1, 1),
        torch.empty(2, 2),
        torch.empty(3, 3),
        torch.empty(4, 4)
    ])