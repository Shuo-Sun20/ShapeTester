import torch
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    "inputs": [torch.randn(3, 2), torch.randn(3, 2)],
    "out": None
}

@dataclass
class InputSpace:
    inputs: List[List[torch.Tensor]] = field(default_factory=lambda: [
        [torch.randn(3, 2), torch.randn(3, 2)],
        [torch.randn(1), torch.randn(1)],
        [torch.randn(3, 1), torch.randn(1, 4)],
        [torch.randn(5), torch.randn(5, 1)],
        [torch.randn(2, 3, 4), torch.randn(2, 1, 4)]
    ])
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.randn(3, 2),
        torch.randn(3, 4),
        torch.randn(5, 5),
        torch.randn(2, 3, 4)
    ])