import torch
from dataclasses import dataclass

# Define a valid test case that can be called via call_func(**valid_test_case)
input_tensor = torch.randn(3, 4, 5)
mask = torch.randint(0, 2, (3, 4, 5), dtype=torch.bool)
dim = 1
valid_test_case = {
    'inputs': [input_tensor, mask],
    'dim': dim,
    'dtype': None,
    'mask': None
}

# Parameters that can affect the output tensor shape (excluding "inputs")
# Only "dim" affects the computational dimension but not the output shape.
# The output shape always matches the input shape.
# Therefore, no parameters except "inputs" affect the shape.
# We'll create an empty InputSpace class accordingly.

@dataclass
class InputSpace:
    # No parameters affect output shape except inputs
    pass