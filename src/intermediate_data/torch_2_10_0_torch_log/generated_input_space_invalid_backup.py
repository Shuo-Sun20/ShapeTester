import torch
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {"inputs": torch.rand(5) * 5}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.tensor([1.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.empty(3, 4)
        ]
    )