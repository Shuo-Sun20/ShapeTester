import torch
from dataclasses import dataclass

# Define example_input as in the provided code
real_part = torch.randn(3, 4)
imag_part = torch.randn(3, 4)
imag_part[0, 0] = 0
imag_part[1, 2] = 0
imag_part[2, 3] = 0
example_input = torch.complex(real_part, imag_part)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": example_input
}

# Task 2-4: Define InputSpace
# torch.isreal only has 'input' parameter (named 'inputs' in call_func)
# There are no additional parameters that affect output shape beyond 'inputs'
# Since we exclude 'inputs' itself, InputSpace is empty

@dataclass
class InputSpace:
    # No fields since torch.isreal has no additional parameters affecting output shape
    pass