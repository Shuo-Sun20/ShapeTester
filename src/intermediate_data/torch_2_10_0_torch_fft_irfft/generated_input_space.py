import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case
real_tensor = torch.randn(5)
T = torch.fft.rfft(real_tensor)
valid_test_case = {
    'inputs': T,
    'n': 5,
    'dim': -1,
    'norm': None,
    'out': None
}

# 2 & 3. Parameters affecting output shape: n, dim (norm and out do not affect shape)

# 4. Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    # n: can be None or positive integer. For discretization:
    #   Boundary: None (default), 1 (min), large value (100)
    #   Typical: 2 (even, small), 3 (odd, small), 4 (even), 5 (odd), 10 (even)
    n: List[Optional[int]] = field(
        default_factory=lambda: [None, 1, 2, 3, 4, 5, 10, 100]
    )
    
    # dim: integer dimension index. For 1D input (size 3 from rfft of length 5):
    #   Valid: 0, -1 (only dimensions for 1D tensor)
    #   Extended for general case: also include -2, 1 for 2D+ tensors
    #   Boundary: -2 (invalid for 1D but valid for 2D), -1, 0, 1
    dim: List[int] = field(
        default_factory=lambda: [-2, -1, 0, 1]
    )

# Example instantiation
var = InputSpace()