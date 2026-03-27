import torch
import torch.nn as nn
from dataclasses import dataclass, field

# 1. Define valid_test_case
torch.manual_seed(42)
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
valid_test_case = {
    'dim': 1,
    'eps': 1e-8,
    'inputs': [input1, input2]
}

# 2-4. Define InputSpace dataclass with discretized parameter values
@dataclass
class InputSpace:
    # Only 'dim' affects the output shape (not 'eps' or 'inputs')
    # For 2D tensor of shape (100, 128):
    # Valid dim values: -2, -1, 0, 1 (where -2=0, -1=1 in this 2D case)
    # We'll include boundary values and typical values for general tensors
    dim: list = field(default_factory=lambda: [
        # Boundary/edge cases
        -4, -3, -2, -1, 0, 1, 2, 3, 4,
        # Additional typical values for higher dimensions
        -5, 5, -6, 6
    ])

# Test instantiation
var = InputSpace()