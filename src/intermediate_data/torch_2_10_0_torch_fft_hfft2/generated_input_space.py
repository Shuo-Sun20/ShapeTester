import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# 1. Define a valid test case
T = torch.rand(10, 9)
t = torch.fft.ihfft2(T)

valid_test_case = {
    'inputs': t,
    's': T.size(),  # (10, 9)
    'dim': (-2, -1),
    'norm': None,
    'out': None
}

# 2. Parameters affecting output shape (except 'inputs'):
#    - s: Signal size in transformed dimensions
#    - dim: Dimensions to be transformed

# 3. & 4. Define InputSpace with discretized value ranges
@dataclass
class InputSpace:
    # s: Can be None, tuple of ints, or tuple with -1
    s: List[Optional[Tuple[int, ...]]] = field(
        default_factory=lambda: [
            None,                     # Default behavior
            (10, 9),                  # Original shape from example
            (8, 8),                   # Smaller even dimensions
            (12, 12),                 # Larger even dimensions  
            (10, -1),                 # No padding in last dimension
            (-1, 9),                  # No padding in first dimension
            (16, 16),                 # Power of 2 (required for half precision)
            (10, 8),                  # Mixed even dimensions
            (6, 12),                  # Different sizes
            (20, 20)                  # Larger boundary case
        ]
    )
    
    # dim: Tuple specifying dimensions to transform
    dim: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (-2, -1),                 # Default: last two dimensions
            (0, 1),                   # First two dimensions
            (-1, -2),                 # Reversed order (valid if input permits)
            (1, 0),                   # Swapped dimensions
            (0, -1),                  # Mixed positive/negative indexing
            (-3, -2),                 # Different negative indices
            (1, 2),                   # Different positive indices
            (-2, 0)                   # Mixed indices
        ]
    )