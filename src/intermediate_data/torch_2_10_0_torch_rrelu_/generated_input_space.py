import torch
from typing import List
from dataclasses import dataclass, field

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4),
    "lower": 0.125,
    "upper": 0.333,
    "training": True
}

# 2. Identify parameters affecting output shape (except "inputs")
# For torch.rrelu_, only the "inputs" parameter affects the output shape.
# The other parameters (lower, upper, training) affect output values but not shape.
# Therefore, there are no parameters besides "inputs" that affect shape.

# 3. Analyze parameter types and construct value spaces
# Since no other parameters affect shape, we only need to handle those for completeness
# in InputSpace definition as per requirements.

# For lower: continuous float in reasonable range (0, 1)
lower_values = [0.001, 0.01, 0.125, 0.25, 0.499]  # Boundary and typical values

# For upper: continuous float in reasonable range (0, 1), must be > lower typically
upper_values = [0.01, 0.125, 0.333, 0.499, 0.999]  # Boundary and typical values

# For training: discrete boolean
training_values = [True, False]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    lower: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.125, 0.25, 0.499])
    upper: List[float] = field(default_factory=lambda: [0.01, 0.125, 0.333, 0.499, 0.999])
    training: List[bool] = field(default_factory=lambda: [True, False])

# Example instantiation
var = InputSpace()