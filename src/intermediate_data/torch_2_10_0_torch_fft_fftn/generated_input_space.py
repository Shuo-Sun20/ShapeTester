from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import torch

# Task 1: Define valid_test_case
x = torch.rand(10, 10, dtype=torch.complex64)
valid_test_case = {
    'inputs': [x],
    's': None,
    'dim': None,
    'norm': None,
    'out': None
}

# Task 2 & 3: Parameters affecting output shape: s and dim
# Discretized value spaces for each parameter

# For s parameter:
# - None: use original sizes
# - Tuple[int]: pad/trim each dimension
# Boundary cases and typical values for a 10x10 input:
#   Original size, larger padding, smaller trimming, mixed, -1 (no padding)
s_values = [
    None,
    (10, 10),      # same size
    (12, 12),      # pad both dimensions
    (8, 8),        # trim both dimensions
    (12, 8),       # pad first, trim second
    (8, 12),       # trim first, pad second
    (10, 12),      # keep first, pad second
    (10, 8),       # keep first, trim second
    (12, 10),      # pad first, keep second
    (8, 10),       # trim first, keep second
    (-1, 10),      # no padding in first, keep second
    (10, -1),      # keep first, no padding in second
    (-1, -1),      # no padding in both dimensions
    (5, 15),       # trim first heavily, pad second heavily
    (15, 5),       # pad first heavily, trim second heavily
    (1, 1),        # minimum trimming
    (20, 20)       # maximum padding (reasonable bound)
]

# For dim parameter:
# - None: all dimensions
# - Tuple[int]: specific dimensions
# Values must be within input tensor's dimensions (0 or 1 for 2D tensor)
dim_values = [
    None,
    (0,),          # transform only first dimension
    (1,),          # transform only second dimension
    (0, 1),        # transform both in order
    (1, 0),        # transform both in reverse order
    (-2, -1),      # negative indexing
    (-1, -2),      # negative indexing reversed
    (-2,),         # single negative dimension
    (-1,),         # single negative dimension
]

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    s: List[Optional[Tuple[Optional[int], ...]]] = field(default_factory=lambda: s_values)
    dim: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: dim_values)

# Example instantiation
var = InputSpace()