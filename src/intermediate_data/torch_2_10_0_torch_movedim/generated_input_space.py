import torch
from dataclasses import dataclass, field
from typing import Union, Tuple

def call_func(inputs, source, destination):
    return torch.movedim(inputs, source, destination)

# 1. Valid test case
valid_test_case = {
    "inputs": torch.randn(3, 2, 1),
    "source": 1,
    "destination": 0
}

# 2-4. InputSpace definition
@dataclass
class InputSpace:
    """
    Contains all parameters affecting output tensor shape (except 'inputs')
    with discretized value ranges (maximum 5 values each)
    """
    # Discretized to boundary values and 5 typical values
    source: list = field(default_factory=lambda: [
        # Single dimension cases
        0, 
        1, 
        2, 
        # Multi-dimension tuples
        (0, 1),
        (1, 2)
    ])
    
    destination: list = field(default_factory=lambda: [
        # Single dimension cases
        0, 
        1, 
        2, 
        # Multi-dimension tuples
        (0, 1),
        (1, 2)
    ])