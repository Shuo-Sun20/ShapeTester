import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(10, 10)],
    "s": None,
    "dim": (-2, -1),
    "norm": None,
    "out": None
}

# 2. Parameters affecting output shape (excluding "inputs"): s, dim
# 3. Value spaces:
#    - s: can be None or tuple of 2 ints (or tuple with -1)
#    - dim: tuple of 2 ints (negative or positive indices)
#    - norm: str or None (doesn't affect shape but included for completeness)

@dataclass
class InputSpace:
    """Dataclass containing parameters that affect output shape of torch.fft.rfft2"""
    
    # Parameter s: Signal size in transformed dimensions
    # Values: None, or tuple of 2 ints where each can be -1, 0, or positive
    # For input size (10,10) as in our example
    s: List[Optional[Tuple[int, int]]] = field(
        default_factory=lambda: [
            None,  # default - use input size
            (-1, -1),  # no padding in both dimensions
            (5, 5),  # trim both dimensions
            (10, 10),  # same as input
            (15, 15),  # pad both dimensions
            (5, 10),  # trim first, keep second
            (10, 15),  # keep first, pad second
            (0, 10),  # boundary: trim first to 0
            (10, 0),  # boundary: trim second to 0
            (-1, 5),  # no padding in first, trim second
            (5, -1),  # trim first, no padding in second
            (20, 20),  # significant padding
            (3, 8),  # asymmetric sizes
        ]
    )
    
    # Parameter dim: Dimensions to transform
    # Values: tuple of 2 distinct ints within tensor dimension range
    # For 2D input tensor
    dim: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (-2, -1),  # default - last two dimensions
            (0, 1),    # same as (-2, -1) for 2D tensor
            (-1, -2),  # reversed order
            (1, 0),    # same as (-1, -2) for 2D tensor
            (-2, 0),   # mixed negative/positive (same as (-2, -2) for 2D)
            (0, -1),   # mixed negative/positive
        ]
    )