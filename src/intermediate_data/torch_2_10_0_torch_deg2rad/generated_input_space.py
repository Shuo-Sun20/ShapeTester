import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define a valid test case
valid_test_case = {
    "inputs": torch.randn(3, 4) * 180.0,
    "out": None
}

# 2. & 3. Parameters that affect output shape (except "inputs"):
#    - "out": Can be None or a tensor with same shape as input
#    Value space: [None] + tensors with different properties

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.tensor([1.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[[1.0]]]),
            torch.empty(0)  # empty tensor
        ]
    )