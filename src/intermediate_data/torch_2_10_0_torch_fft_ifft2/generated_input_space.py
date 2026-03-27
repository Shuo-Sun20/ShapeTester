import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# 1. Define valid_test_case
input_tensor = torch.rand(10, 10, dtype=torch.complex64)
valid_test_case = {
    'inputs': input_tensor,
    's': None,
    'dim': (-2, -1),
    'norm': None,
    'out': None
}

# 2. Parameters affecting output shape (except "inputs"): s, dim

# 3. Discretized value spaces
s_values = [
    None,
    (-1, -1),      # no padding in both dimensions
    (8, 8),        # smaller than input (trimming)
    (10, 10),      # same as input (no change)
    (12, 12),      # larger than input (padding)
    (8, 12),       # mixed sizes
    (16, 16)       # power of 2 (common for FFT)
]

dim_values = [
    (-2, -1),      # default - last two dimensions
    (0, 1),        # first two dimensions
    (1, 0),        # reversed order (should be supported)
    (0, -1),       # mixed positive/negative indices
    (-1, -2),      # reversed negative indices
    (1, 2),        # middle dimensions (for >=3D tensors)
    (0, -2)        # first and second-to-last
]

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    s: List[Optional[Tuple[int, ...]]] = field(
        default_factory=lambda: s_values
    )
    dim: List[Tuple[int, ...]] = field(
        default_factory=lambda: dim_values
    )