import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define a valid test case
tensor1 = torch.randn(3, 1)
tensor2 = torch.randn(1, 4)
valid_test_case = {
    "inputs": [tensor1, tensor2],
    "out": None
}

# 2, 3, 4. Define InputSpace dataclass with all parameters affecting output shape
@dataclass
class InputSpace:
    # 'out' is the only parameter other than 'inputs' that can affect output shape
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([]),
        torch.zeros(3, 4),
        torch.ones(3, 4),
        torch.randn(3, 4)
    ])