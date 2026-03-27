import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

# 1. Define a valid test case dictionary
valid_test_case = {
    "inputs": torch.randn(20, 16),
    "p": 0.2,
    "inplace": False
}

# 2. Parameters affecting output shape: None (only "inputs" affects shape)
# 3. Value spaces for all parameters:
#    - p: float in [0, 1), discretized to boundary/typical values
#    - inplace: boolean, discrete values [True, False]

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape.
    Since only 'inputs' affects shape and it's excluded, this class contains
    all other parameters with their discretized value spaces.
    """
    # p: probability of dropout, float in [0, 1)
    # Boundary values: 0.0, 0.999..., plus typical values in between
    p: List[float] = field(default_factory=lambda: [
        0.0,          # Lower bound (no dropout)
        0.1,          # Low dropout
        0.2,          # Medium-low (from valid_test_case)
        0.3,          # Medium
        0.5,          # Default value, medium-high
        0.7,          # High
        0.8,          # Very high
        0.99,         # Near upper bound (practically 1)
        # Note: p=1.0 is not valid as it would drop all elements
    ])
    
    # inplace: boolean flag for inplace operation
    inplace: List[bool] = field(default_factory=lambda: [
        False,        # Non-inplace operation (default)
        True,         # Inplace operation
    ])

# Example instantiation
var = InputSpace()