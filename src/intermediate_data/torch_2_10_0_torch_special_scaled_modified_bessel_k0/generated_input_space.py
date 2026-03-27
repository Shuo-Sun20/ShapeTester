import torch
from dataclasses import dataclass
from typing import Optional

def call_func(inputs, out=None):
    return torch.special.scaled_modified_bessel_k0(inputs, out=out)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4, dtype=torch.float32),
    "out": None
}

# 2 & 3. Identify parameters affecting output shape and construct value spaces
# The only parameter besides "inputs" is "out", which must be a tensor with the same shape as inputs or None
# For discrete parameter "out", possible values are None and tensors matching input shape

@dataclass
class InputSpace:
    """Dataclass containing parameters that affect output tensor shape"""
    out: list[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        # Create a base input tensor for generating out tensors
        base_input = torch.randn(3, 4, dtype=torch.float32)
        
        # Discrete value space for "out" parameter
        # Possible values: None, and tensors with correct shape
        self.out = [
            None,  # Default case
            torch.empty_like(base_input),  # Empty tensor with correct shape
            torch.zeros_like(base_input),  # Zero tensor with correct shape
            torch.ones_like(base_input),  # Ones tensor with correct shape
            torch.full_like(base_input, 0.5),  # Tensor with specific value
            torch.randn_like(base_input)  # Random tensor with correct shape
        ]