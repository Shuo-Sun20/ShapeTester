import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [torch.tensor([[-3, -2, -1], [0, 1, 2]])],
    'dim': 1,
    'keepdim': False,
    'dtype': None,
    'mask': torch.tensor([[True, False, True], [False, False, False]])
}

# 2 & 3. Parameters affecting output shape and their value spaces
# - dim: affects which dimension is reduced
# - keepdim: affects whether reduced dimension is kept
# Note: dtype and mask don't affect output shape

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # dim: integer dimension index, can be negative
    # For 2D input (example case), valid dim values are [-2, -1, 0, 1]
    # We'll cover negative indexing, valid positive indices, and boundary cases
    dim: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])
    
    # keepdim: boolean flag
    keepdim: List[bool] = field(default_factory=lambda: [True, False])