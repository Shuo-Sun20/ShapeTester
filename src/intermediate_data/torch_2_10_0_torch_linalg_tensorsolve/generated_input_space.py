import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List

def call_func(inputs, dims=None, out=None):
    A, B = inputs[0], inputs[1]
    return torch.linalg.tensorsolve(A, B, dims=dims, out=out)

# Valid test case (from documentation example with dims)
A = torch.randn(6, 4, 4, 3, 2)
B = torch.randn(4, 3, 2)
valid_test_case = {
    'inputs': [A, B],
    'dims': (0, 2),
    'out': None
}

@dataclass
class InputSpace:
    """Class containing all parameters affecting output tensor shape"""
    dims: List[Optional[Tuple[int, ...]]] = None
    
    def __post_init__(self):
        if self.dims is None:
            # Discretized value space for dims parameter
            # 1. Boundary/edge cases
            # 2. Typical valid values (must be valid for A.shape=(6,4,4,3,2))
            self.dims = [
                None,  # No dimensions moved
                (),    # Empty tuple
                (0,),  # Single dimension
                (0, 2),  # Valid case from documentation
                (0, 1),  # Another valid pair
                (1, 2),  # Another valid pair
                (0, 1, 2),  # Three dimensions
                (0, 2, 1),  # Different order
                (3, 4),  # Last two dimensions
                (0, 3),  # Mixed dimensions
            ]

# Example instantiation
var = InputSpace()