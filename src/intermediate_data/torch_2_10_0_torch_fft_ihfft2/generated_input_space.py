import torch
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# 1. Define a valid test case
valid_test_case = {
    "inputs": [torch.rand(10, 10)],
    "s": None,
    "dim": (-2, -1),
    "norm": None,
    "out": None
}

# 2. Parameters affecting output shape: s, dim
# 3. & 4. Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    """
    Dataclass containing parameters that affect output shape of torch.fft.ihfft2.
    All values are discretized to cover typical legal scenarios.
    """
    s: List[Optional[Tuple[Optional[int], ...]]] = field(default_factory=lambda: [
        None,
        (-1, -1),
        (8, 8),      # even dimensions
        (9, 9),      # odd dimensions  
        (10, 10),    # same as input (from valid_test_case)
        (12, 12),    # larger than input (padding)
        (6, 6),      # smaller than input (trimming)
        (10, 12),    # mixed dimensions
        (12, 10),    # reversed mixed dimensions
        (16, 16),    # power of 2
        (16, 8),     # mixed with power of 2
        (8, -1),     # partial no-padding
        (-1, 8)      # reversed partial no-padding
    ])
    
    dim: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (-2, -1),     # default
        (0, 1),       # positive indices
        (1, 0),       # reversed order
        (0, -1),      # mixed positive/negative
        (-1, -2),     # reversed default
        (0,),         # single dimension (will be extended by ihfft2)
        (1,),         # single dimension different axis
        (-1,),        # single negative dimension
    ])

# Test instantiation
var = InputSpace()