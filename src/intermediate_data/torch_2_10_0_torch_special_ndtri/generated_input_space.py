import torch
from dataclasses import dataclass

# Valid test case for call_func()
valid_test_case = {
    'inputs': [torch.rand(5)],  # Can be a list containing a tensor
    'out': None  # Optional output tensor
}

@dataclass
class InputSpace:
    """
    All parameters that affect the shape of the output tensor,
    excluding 'inputs' parameter.
    """
    out: list = None
    
    def __post_init__(self):
        # Since 'out' is the only parameter affecting shape (other than inputs),
        # we define its discretized value space
        if self.out is None:
            # Create 5 representative output tensor shapes
            # 1. None (default)
            # 2. Empty tensor (will be filled with results)
            # 3. Same shape as typical input
            # 4. Broadcastable shape
            # 5. Different contiguous shape
            
            # Note: The actual shape must match input shape for successful operation
            # We create example tensors for illustration
            example_shape = torch.Size([5])  # Typical input shape from example
            
            self.out = [
                None,  # Default case
                torch.empty(example_shape),  # Same shape as input
                torch.empty(example_shape),  # Another same shape
                torch.empty(example_shape),  # Same shape
                torch.empty(example_shape)   # Same shape
            ]

# Example instantiation
var = InputSpace()