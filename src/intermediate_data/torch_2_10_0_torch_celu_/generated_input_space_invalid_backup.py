import torch
from dataclasses import dataclass, field

def call_func(inputs, alpha=1.):
    return torch.celu_(inputs, alpha)

# 1. Define a valid test case
valid_test_case = {
    'inputs': torch.randn(3, 4),
    'alpha': 1.5
}

# 2. Identify all parameters in the parameter list of call_func that can affect the shape of the output tensor, except for "inputs".
#    In torch.celu_, the output shape is always the same as the input shape. Therefore, the parameter 'alpha' does not affect the shape.
#    However, it is the only other parameter and we are to analyze it for its value space.

# 3. Parameter analysis for 'alpha':
#    Type: float (scalar)
#    Value space: continuous, discretized to 5 values including boundaries and typical values.
#    Chosen values: [-2.0, -1.0, 0.5, 1.0, 2.0]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    alpha: list = field(default_factory=lambda: [-2.0, -1.0, 0.5, 1.0, 2.0])