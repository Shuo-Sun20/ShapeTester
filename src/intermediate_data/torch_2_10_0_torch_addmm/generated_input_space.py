import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define a valid test case
valid_test_case = {
    "inputs": [torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)],
    "out_dtype": None,
    "beta": 1,
    "alpha": 1,
    "out": None
}

# Task 2 & 3: Identify parameters affecting output shape and construct value spaces
# Only 'out' parameter can affect output shape, as it must match the computed shape

# Task 4: Define InputSpace class
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.randn(2, 3),  # Matching shape for (2,3) x (3,3) -> (2,3)
        torch.randn(2, 3),  # Another valid output tensor
        torch.randn(2, 3),  # Third option
        torch.randn(2, 3)   # Fourth option
    ])