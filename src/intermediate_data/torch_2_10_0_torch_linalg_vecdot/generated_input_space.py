import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case
def call_func(inputs, dim=-1, out=None):
    x, y = inputs
    return torch.linalg.vecdot(x, y, dim=dim, out=out)

v1 = torch.randn(3, 2)
v2 = torch.randn(3, 2)
valid_test_case = {
    'inputs': [v1, v2],
    'dim': -1,
    'out': None
}

# 2. Parameters affecting output shape (excluding "inputs"): dim
# 3. Value space for dim:
#    - Type: int
#    - Legal range: [-rank, rank-1] where rank is tensor dimension
#    - Discrete parameter, list all typical values including boundaries
#    - For tensors of rank up to 5, we cover:
#      * Boundary negatives: -5, -4, -3, -2, -1
#      * Boundary positives: 0, 1, 2, 3, 4
#      * Default value: -1 (included)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # dim affects output shape by determining which dimension is reduced
    dim: List[int] = field(default_factory=lambda: [
        -5,  # Max negative boundary (for rank 5)
        -4,  # Typical negative
        -3,  # Typical negative
        -2,  # Typical negative
        -1,  # Default value & typical negative
        0,   # First dimension
        1,   # Second dimension
        2,   # Third dimension
        3,   # Fourth dimension
        4    # Max positive boundary (for rank 5)
    ])

# Example instantiation
var = InputSpace()