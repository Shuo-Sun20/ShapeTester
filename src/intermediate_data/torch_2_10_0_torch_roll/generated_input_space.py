import torch
from dataclasses import dataclass, field

def call_func(inputs, shifts, dims=None):
    return torch.roll(inputs, shifts, dims)

# Task 1: Valid test case
valid_test_case = {
    'inputs': torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
    'shifts': 1,
    'dims': 0
}

# Tasks 2-4: InputSpace definition
@dataclass
class InputSpace:
    # shifts parameter value space (discrete integer values)
    shifts: list = field(default_factory=lambda: [
        -10,        # Large negative shift
        -1,         # Typical negative shift
        0,          # No shift (boundary)
        1,          # Typical positive shift (included from valid_test_case)
        10,         # Large positive shift
        (2, -3),    # Tuple shifts (mixed directions)
        (-5, 4),    # Tuple shifts (negative and positive)
        (0, 0),     # Tuple of zeros
        (1, 1),     # Equal tuple shifts
        (3, -2, 1)  # 3D tuple example
    ])
    
    # dims parameter value space (discrete dimension specifications)
    dims: list = field(default_factory=lambda: [
        None,       # Flatten and roll (boundary)
        0,          # Single dimension 0 (included from valid_test_case)
        1,          # Single dimension 1
        -1,         # Negative dimension (last dimension)
        (0, 1),     # Two dimensions
        (1, 0),     # Reversed dimension order
        (-1, -2),   # Negative dimensions
        (0,),       # Single-element tuple
        (0, 1, 2),  # Three dimensions (for 3D+ tensors)
        (-2, -1)    # Last two dimensions
    ])