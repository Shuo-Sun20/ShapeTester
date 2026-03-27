import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

def call_func(inputs, dim, unbiased, keepdim=False, dtype=None, mask=None):
    return torch.masked.var(input=inputs, dim=dim, unbiased=unbiased, keepdim=keepdim, dtype=dtype, mask=mask)

# Generate random tensors for input and mask (same as example)
torch.manual_seed(42)
inputs = torch.randn(3, 4, 5)
mask = torch.randint(0, 2, (3, 4, 5), dtype=torch.bool)

# 1. Define valid_test_case
valid_test_case = {
    'inputs': inputs,
    'dim': 1,
    'unbiased': False,
    'keepdim': True,
    'dtype': None,
    'mask': mask
}

# 2. Parameters affecting output shape (excluding inputs): dim, keepdim
# 3. Value spaces for these parameters

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # dim can be int, tuple of ints, or None
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,           # reduce all dimensions
            0,              # reduce dimension 0
            1,              # reduce dimension 1 (from valid_test_case)
            2,              # reduce dimension 2
            (0, 1),         # reduce dimensions 0 and 1
            (0, 2),         # reduce dimensions 0 and 2
            (1, 2),         # reduce dimensions 1 and 2
            (0, 1, 2),      # reduce all dimensions (alternative to None)
            -1,             # reduce last dimension
            -2,             # reduce second last dimension
            (-1, -2),       # reduce last two dimensions
            (0, -1),        # reduce first and last dimensions
        ]
    )
    # keepdim is boolean
    keepdim: List[bool] = field(default_factory=lambda: [True, False])