import torch
from dataclasses import dataclass, field
from typing import List

# 1. Valid test case
valid_test_case = {"inputs": torch.randn(5, 3)}

# 2-4. InputSpace dataclass
@dataclass
class InputSpace:
    """
    torch.positive only has one parameter 'inputs' which affects output shape.
    The shape of output equals the shape of inputs.
    """
    inputs: List[torch.Tensor] = field(
        default_factory=lambda: [
            torch.randn(()),        # 0-d scalar
            torch.randn(3),         # 1-d vector
            torch.randn(2, 3),      # 2-d matrix
            torch.randn(2, 3, 4),   # 3-d tensor
            torch.randn(2, 3, 4, 5) # 4-d tensor
        ]
    )