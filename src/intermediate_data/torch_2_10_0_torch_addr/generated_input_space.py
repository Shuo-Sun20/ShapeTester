import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List
import math

# 1. Define a valid test case
vec1 = torch.randn(3)
vec2 = torch.randn(2)
input_matrix = torch.randn(3, 2)

valid_test_case = {
    "inputs": [input_matrix, vec1, vec2],
    "beta": 1.0,
    "alpha": 1.0,
    "out": None
}

# 2. & 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only 'out' parameter can affect output shape through pre-allocation
    # Values: None for no pre-allocation, or tensor of correct shape
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,  # No pre-allocation
            torch.empty(3, 2),  # Correct shape
            torch.zeros(3, 2),  # Correct shape with zeros
            torch.full((3, 2), float('nan')),  # Correct shape with NaN
            torch.full((3, 2), float('inf'))   # Correct shape with Inf
        ]
    )