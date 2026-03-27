import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union

# 1. Define a valid test case
valid_test_case = {
    'inputs': torch.randn(4, 4),
    'dim': 1,
    'keepdim': False,
    'other': None,
    'out': None
}

# 2. & 3. Parameters that affect output shape and their value spaces:
# - dim: affects shape when specified (reduces dimension)
# - keepdim: affects shape when dim is specified
# - other: affects shape through broadcasting when provided

@dataclass
class InputSpace:
    dim: List[Optional[int]] = field(default_factory=lambda: [
        None,  # reduce all dimensions
        0,     # reduce first dimension
        1,     # reduce second dimension
        -1,    # reduce last dimension
        -2     # reduce second-last dimension
    ])
    
    keepdim: List[bool] = field(default_factory=lambda: [
        False,  # squeeze reduced dimension
        True    # keep reduced dimension as size 1
    ])
    
    other: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,                     # single tensor mode
        torch.randn(4, 4),       # same shape
        torch.randn(4, 1),       # broadcastable shape
        torch.randn(1, 4),       # broadcastable shape
        torch.randn(1, 1)        # scalar broadcast
    ])