from dataclasses import dataclass
import torch

def call_func(inputs, out=None):
    return torch.special.spherical_bessel_j0(inputs, out=out)

example_input = torch.randn(3, 4)
example_output = call_func(example_input)

# 1. Define a valid test case
valid_test_case = {
    "inputs": torch.tensor([0.0, 1.0, 2.0]),  # Valid input tensor
    "out": None  # Optional out parameter
}

# 2. Identify parameters affecting output shape (except "inputs")
# Only "out" parameter affects output shape if provided
# 3. Construct value spaces for these parameters

@dataclass
class InputSpace:
    # Only parameter that affects output shape (besides inputs) is 'out'
    # Discretized value space for 'out' parameter
    out: list = None
    
    def __post_init__(self):
        # Define value space for 'out' parameter
        # Discrete values: None, and tensors of various shapes
        if self.out is None:
            # Create 6 typical values for 'out' parameter
            # 1. None (default)
            # 2-6: Tensors of various shapes (must match input shape when used)
            self.out = [
                None,  # Default case
                torch.zeros(3),  # Same shape as valid_test_case input
                torch.empty(3),  # Empty tensor with same shape
                torch.zeros(2, 3),  # Different shape 1
                torch.zeros(1, 2, 3),  # Different shape 2
                torch.zeros(3, 1)  # Different shape 3
            ]

# Create instance
var = InputSpace()