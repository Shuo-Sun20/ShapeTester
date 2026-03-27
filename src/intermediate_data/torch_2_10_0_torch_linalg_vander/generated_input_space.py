import torch
from dataclasses import dataclass, field
from typing import Optional

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [torch.tensor([1, 2, 3, 5])],
    "N": 3
}

# 3. Define discretized value spaces
N_values = [
    None,  # Default case (N = x.size(-1))
    0,     # Minimum valid value (empty matrix)
    1,     # Boundary: single column of ones
    2,     # Small non-trivial value
    3,     # Example value from valid_test_case
    4,     # Equal to x.size(-1) for 1D input
    5,     # Larger than x.size(-1)
    10,    # Typical larger value
    100,   # Large value
]

# 2. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    N: list = field(default_factory=lambda: N_values)