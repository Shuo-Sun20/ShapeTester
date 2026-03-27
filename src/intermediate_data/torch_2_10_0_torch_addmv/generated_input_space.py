import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [
        torch.randn(2),    # input_tensor (size n=2)
        torch.randn(2, 3), # mat (size n×m=2×3)
        torch.randn(3)     # vec (size m=3)
    ],
    'beta': 1.0,
    'alpha': 1.0,
    'out': None
}

# 2. Parameters affecting output shape (except "inputs"): out

@dataclass
class InputSpace:
    # Parameters affecting output shape (except "inputs")
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros(2),
        torch.ones(2),
        torch.randn(2),
        torch.full((2,), 2.0)
    ])