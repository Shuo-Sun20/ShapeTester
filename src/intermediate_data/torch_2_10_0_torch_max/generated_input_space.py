import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Define a valid test case
valid_test_case = {
    'inputs': torch.randn(4, 4),
    'dim': 1,
    'keepdim': False,
    'out': None
}

# 2, 3 & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape (excluding inputs)
    dim: List[Optional[int]] = field(
        default_factory=lambda: [
            None,       # Global reduction
            0,          # First dimension
            1,          # Second dimension
            -1,         # Last dimension
            -2          # Second last dimension
        ]
    )
    keepdim: List[bool] = field(
        default_factory=lambda: [
            True,
            False
        ]
    )