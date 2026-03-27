import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(4, 6, 8, dtype=torch.float32)],  # Example input tensor
    's': (6, 8),  # Signal size in transformed dimensions
    'dim': (1, 2),  # Dimensions to transform
    'norm': 'ortho',  # Normalization mode
    'out': None  # Output tensor (optional)
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape of torch.fft.rfftn,
    along with their discretized value spaces.
    """
    s: List[Optional[Tuple[int, ...]]]  # Signal size
    dim: List[Optional[Tuple[int, ...]]]  # Dimensions to transform

    def __init__(self):
        # Discretized value space for 's'
        self.s = [
            None,  # Default: use input size
            (-1,),  # No padding in first dimension
            (4,),  # Trim/pad to 4 in first dimension
            (8,),  # Trim/pad to 8 in first dimension
            (16,),  # Trim/pad to 16 in first dimension
            (32,),  # Large size for boundary testing
            (-1, -1),  # No padding in two dimensions
            (4, 6),  # Mixed sizes in two dimensions
            (6, 8),  # Included from valid_test_case
            (8, 12),  # Larger sizes in two dimensions
            (16, 32),  # Large sizes for boundary testing
            (-1, -1, -1),  # No padding in three dimensions
            (4, 6, 8),  # Three-dimensional sizes
            (8, 12, 16),  # Larger three-dimensional sizes
        ]

        # Discretized value space for 'dim'
        # Assuming input tensor has 3 dimensions (0, 1, 2)
        self.dim = [
            None,  # Default: all dimensions
            (0,),  # Transform only dimension 0
            (1,),  # Transform only dimension 1
            (2,),  # Transform only dimension 2
            (0, 1),  # Transform two dimensions
            (0, 2),  # Transform two dimensions
            (1, 2),  # Included from valid_test_case
            (0, 1, 2),  # Transform all dimensions
            (-1,),  # Negative index: last dimension
            (-2,),  # Negative index: second last dimension
            (-3,),  # Negative index: third last dimension
            (-2, -1),  # Negative indices for two dimensions
        ]

# Example instantiation
var = InputSpace()