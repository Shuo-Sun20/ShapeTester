import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

# 1. Define valid_test_case
t = torch.rand(10, 9)
T = torch.fft.rfftn(t)

valid_test_case = {
    'inputs': T,
    's': t.size(),  # torch.Size([10, 9])
    'dim': None,
    'norm': None,
    'out': None
}

# 2 & 3. Parameters affecting output shape (excluding "inputs"): s, dim
# Their value spaces:

# s: can be None, tuple of ints, or tuple with -1 for no padding
s_values = [
    None,  # default behavior
    (10, 9),  # original shape (from valid_test_case)
    (8, 8),  # crop in both dimensions
    (12, 12),  # pad in both dimensions
    (10, -1),  # no padding in last dimension
    (8, -1),  # crop first, no pad last
    (-1, 9),  # no pad first, original last
    (5, 5),  # small crop
    (16, 16),  # large pad
    (10, 16),  # mixed
    (16, 9),  # mixed
]

# dim: can be None, or tuple specifying dimensions to transform
# For 2D input with shape (10, 5) after rfftn
dim_values = [
    None,  # all dimensions
    (0, 1),  # both dimensions (explicit)
    (1, 0),  # reversed order
    (0,),  # only first dimension
    (1,),  # only second dimension
    (-2, -1),  # negative indices
    (-1, -2),  # reversed negative indices
    (0, -1),  # mixed indices
]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    s: List[Optional[Tuple[int, ...]]] = field(
        default_factory=lambda: s_values
    )
    dim: List[Optional[Tuple[int, ...]]] = field(
        default_factory=lambda: dim_values
    )

# Test instantiation
var = InputSpace()