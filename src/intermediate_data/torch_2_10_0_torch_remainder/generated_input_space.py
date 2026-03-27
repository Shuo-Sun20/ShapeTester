import torch
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    "inputs": [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 2.0, 2.0])],
    "out": None
}

@dataclass
class InputSpace:
    out: Optional[List[Optional[torch.Tensor]]] = field(
        default_factory=lambda: [
            None,
            torch.tensor([0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 0.0])
        ]
    )