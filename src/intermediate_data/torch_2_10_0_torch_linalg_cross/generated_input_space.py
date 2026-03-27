from dataclasses import dataclass, field
from typing import Optional, List
import torch

def call_func(inputs, dim=-1, out=None):
    input_tensor, other_tensor = inputs[0], inputs[1]
    return torch.linalg.cross(input=input_tensor, other=other_tensor, dim=dim, out=out)

# 1. Define valid_test_case
tensor_a = torch.randn(4, 3)
tensor_b = torch.randn(4, 3)
valid_test_case = {
    'inputs': [tensor_a, tensor_b],
    'dim': -1,
    'out': None
}

# 2. & 3. Parameters affecting output shape (except 'inputs'): dim
# dim can be any integer representing a valid dimension index for the input tensors.
# Since cross requires dimension of size 3, typical values range from -rank to rank-1
# where rank is typically 2 (matrix) or higher (batched).
# We include boundary values and typical values covering different scenarios.

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # dim: dimension along which cross product is computed
    # Values include negative indices, positive indices, and boundary cases
    # For typical 2D input (batch, 3), valid dim values are [-2, -1, 0, 1]
    # We extend to cover higher rank scenarios and include at least 5 typical values
    dim: List[int] = field(default_factory=lambda: [
        -3,  # Boundary: negative index for 3D+ tensors
        -2,  # Common: second-to-last dimension for 3D tensors
        -1,  # Default: last dimension
        0,   # First dimension for batched 2D/3D tensors
        1,   # Second dimension for batched 2D tensors
        2    # Third dimension for 3D+ tensors (if applicable)
    ])
    # Note: 'out' parameter does not affect output shape, so it's not included here

# Example instantiation
var = InputSpace()