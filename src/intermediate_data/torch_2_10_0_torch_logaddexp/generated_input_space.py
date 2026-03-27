import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union
import math

valid_test_case = {
    "inputs": [torch.randn(3, 2), torch.randn(3, 2)],
    "out": None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty((1,)),
        torch.empty((3,)),
        torch.empty((5,)),
        torch.empty((2, 3)),
        torch.empty((3, 2)),
        torch.empty((1, 2, 3)),
        torch.empty((2, 3, 4)),
        torch.empty((4, 1, 5, 2))
    ])