import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple

# The provided call_func implementation
def call_func(inputs, dim=None, keepdim=False, dtype=None, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    if dim is None:
        return torch.mean(input_tensor, dtype=dtype)
    else:
        return torch.mean(input_tensor, dim=dim, keepdim=keepdim, dtype=dtype, out=out)

# 1. Valid test case
example_input = torch.randn(2, 3)
valid_test_case = {
    "inputs": example_input,
    "dim": 1,
    "keepdim": False,
    "dtype": None,
    "out": None
}

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # Parameters affecting output shape:
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            # None case (reduces all dimensions)
            None,
            # Single dimension cases
            0, 1, 2, -1, -2,
            # Multi-dimension tuples
            (0, 1), (0, 2), (1, 2), (0, -1), (-2, -1),
            # Edge cases
            (),  # Empty tuple (no reduction)
            (0,), (1,), (2,),  # Single element tuples
            (1, 0), (2, 1, 0),  # Different orderings
            (-1, 0),  # Mixed positive/negative
        ]
    )
    keepdim: List[bool] = field(
        default_factory=lambda: [False, True]
    )