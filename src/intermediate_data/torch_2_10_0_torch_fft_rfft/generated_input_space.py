import torch
import torch.fft
from dataclasses import dataclass
from typing import List, Optional

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.rfft(input=inputs, n=n, dim=dim, norm=norm, out=out)

# 1. Define the valid_test_case dictionary
valid_test_case = {
    "inputs": torch.randn(8),
    "n": None,
    "dim": -1,
    "norm": None,
    "out": None
}

# 2 & 3: Parameters affecting output shape: n, dim
# 4. Define the InputSpace dataclass
@dataclass
class InputSpace:
    # n: Can be None or positive integer
    # Include boundary and typical values:
    # - None (default)
    # - n smaller than input length (trimming)
    # - n equal to input length (no change)
    # - n larger than input length (zero-padding)
    # - Powers of 2 (special requirement for half precision)
    # - Non-powers of 2 (general case)
    n: List[Optional[int]] = None
    dim: List[int] = None
    
    def __post_init__(self):
        # Set default value spaces if not provided
        if self.n is None:
            # For input tensor with size 8 in example
            self.n = [
                None,           # default, no padding/trimming
                0,              # boundary (invalid, but included for testing)
                1,              # minimal length
                2,              # small power of 2
                3,              # small non-power of 2
                4,              # medium power of 2
                7,              # medium non-power of 2
                8,              # equal to input length
                9,              # just above input length
                12,             # above input length, non-power of 2
                16,             # above input length, power of 2
                32              # far above input length
            ]
        if self.dim is None:
            # For input tensor with shape (8,) in example
            # dim can be 0 or -1, but let's consider multi-dimensional cases
            # by allowing any valid dimension index for a 1D tensor
            self.dim = [-2, -1, 0, 1]  # Includes invalid (-2) to test boundaries

# Example instantiation
var = InputSpace()