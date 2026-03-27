import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

# Original function definition
def call_func(inputs, s=None, dim=None, norm=None, out=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    return torch.fft.ifftn(input_tensor, s=s, dim=dim, norm=norm, out=out)

# Valid test case
x = torch.rand(10, 10, dtype=torch.complex64)
valid_test_case = {
    'inputs': x,
    's': None,
    'dim': None,
    'norm': None,
    'out': None
}

@dataclass
class InputSpace:
    """Dataclass containing parameters that affect output shape with discretized value spaces"""
    
    # Parameter s: affects output shape by padding/trimming dimensions
    # Type: Optional[Tuple[int, ...]] - Tuple of integers or None
    # Discretized values covering all legal scenarios:
    # 1. None (default)
    # 2. Same shape as input
    # 3. Smaller than input (trimming)
    # 4. Larger than input (padding)
    # 5. Mixed -1 (no padding) with other values
    # 6. Single dimension specification
    s: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: [
        None,
        (10, 10),      # Same as input
        (5, 5),        # Trim both dimensions
        (16, 16),      # Pad to power of 2 (CUDA requirement)
        (8, 12),       # Mixed padding/trimming
        (10, -1),      # -1 for no padding in dimension
        (-1, 20),      # -1 in first dimension
        (8,),          # Single dimension (last len(s) dims)
        (16, 8, 4),    # 3D case
        (1, 10),       # Boundary: minimal size
        (100, 100)     # Large padding
    ])
    
    # Parameter dim: affects which dimensions are transformed
    # Type: Optional[Tuple[int, ...]] - Tuple of dimension indices or None
    # Discretized values covering all legal scenarios:
    # 1. None (default - all dimensions or last len(s) dims)
    # 2. Single dimension
    # 3. Multiple dimensions in order
    # 4. Multiple dimensions out of order
    # 5. Negative indices (wrapping)
    # 6. Empty tuple
    dim: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: [
        None,
        (0,),          # Single dimension
        (1,),          # Another single dimension
        (0, 1),        # Both dimensions in order
        (1, 0),        # Both dimensions reversed
        (-1,),         # Negative index (last dimension)
        (-2, -1),      # Negative indices for last two
        (0,),          # For 1D transform
        (0, 2),        # Non-consecutive dimensions
        (0, 1, 2),     # 3D case
        ()             # Empty tuple (no dimensions transformed)
    ])