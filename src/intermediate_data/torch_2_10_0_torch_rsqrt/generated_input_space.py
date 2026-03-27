import torch
from dataclasses import dataclass

# Task 1: Define a valid test case
valid_test_case = {
    'inputs': torch.rand(4) + 0.1,
    'out': None
}

# Task 2 & 3 & 4: Define InputSpace class with parameters that affect output shape
@dataclass
class InputSpace:
    # The only parameter other than 'inputs' is 'out', which must be None or a tensor matching input shape
    out: list = None

    def __post_init__(self):
        # Define the value space for 'out' parameter
        # Discrete values: None (default), and tensors with valid shapes (same as input)
        # For demonstration, we create sample tensors with typical shapes and data types
        if self.out is None:
            # Create a list of possible values for 'out'
            # 1. None (default)
            # 2-6. Example tensors of various shapes (all must match input shape in practice)
            # Note: Since 'out' must match input shape, we demonstrate with a fixed input shape of (4,)
            sample_tensor = torch.randn(4)
            self.out = [
                None,  # No output tensor provided
                torch.empty_like(sample_tensor),  # Same shape and dtype as input
                torch.zeros_like(sample_tensor),  # Same shape, zeros
                torch.ones_like(sample_tensor) * 0.5,  # Same shape, filled with 0.5
                sample_tensor.clone(),  # Clone of input (valid but unusual)
                torch.full_like(sample_tensor, 2.0)  # Same shape, filled with 2.0
            ]