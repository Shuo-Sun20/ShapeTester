import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Define the call_func as given
def call_func(inputs, dim, dtype=None):
    return torch.special.log_softmax(input=inputs, dim=dim, dtype=dtype)

# Generate a random tensor for input
input_tensor = torch.randn(3, 4)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": input_tensor,
    "dim": 1,
    "dtype": None
}

# 2. Parameters affecting output shape (except inputs): only "dim"
#    "dtype" does not affect shape, only data type.

# 3. Value space for "dim":
#    - Discrete parameter (int)
#    - Can be positive (0 to ndim-1) or negative (-ndim to -1)
#    - For 2D input (3,4), possible values: -2, -1, 0, 1
#    - For generalization, we consider typical values for any tensor:
#      Positive: 0, 1, 2, 3, 4 (covers up to 5D tensors)
#      Negative: -5, -4, -3, -2, -1 (matches positive range)
#      Include the valid_test_case value (1)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    dim: List[int] = field(default_factory=lambda: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    # dtype is not included as it does not affect shape

# Example instantiation
var = InputSpace()