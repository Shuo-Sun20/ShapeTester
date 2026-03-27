import torch
from dataclasses import dataclass, field
from typing import List, Union

def call_func(inputs, dims):
    return torch.flip(inputs, dims)

# Task 1: Valid test case
valid_test_case = {
    'inputs': torch.randn(2, 3, 4),
    'dims': [0, 1]
}

# Task 4: InputSpace class
@dataclass
class InputSpace:
    # Only dims parameter affects the output tensor
    # For a 3D tensor example with shape (2,3,4), valid dims are [-3,-2,-1,0,1,2]
    # Discretized to 5 representative values
    dims: List[List[int]] = field(default_factory=lambda: [
        [],          # empty list (no flipping)
        [0],         # single positive dimension
        [0, 1],      # multiple dimensions (as in example)
        [-1],        # single negative dimension
        [0, -1]      # mixed positive and negative dimensions
    ])