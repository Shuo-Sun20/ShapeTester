import torch
from dataclasses import dataclass, field
from typing import List

# 1. Valid test case
valid_test_case = {"inputs": torch.randn(3, 3, dtype=torch.complex64)}

# 4. InputSpace definition
@dataclass
class InputSpace:
    inputs: List[torch.Tensor] = field(default_factory=lambda: [
        torch.randn(2, 3, dtype=torch.complex64),
        torch.randn(3, 4, dtype=torch.complex64),
        torch.randn(4, 2, dtype=torch.complex64),
        torch.randn(5, 1, dtype=torch.complex64),
        torch.randn(1, 6, dtype=torch.complex64)
    ])