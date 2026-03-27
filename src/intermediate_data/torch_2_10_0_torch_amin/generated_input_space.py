import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple

# 1. Valid test case dictionary
valid_test_case = {
    'inputs': [torch.randn(4, 4)],
    'dim': 1,
    'keepdim': False,
    'out': None
}

# 3-4. InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    # dim: int or tuple of ints (discrete - limited to 5 values)
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [None, 0, 1, (0, 1), -1]
    )
    
    # keepdim: boolean (discrete)
    keepdim: List[bool] = field(default_factory=lambda: [True, False])