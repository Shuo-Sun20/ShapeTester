from dataclasses import dataclass
import torch

def call_func(inputs, alpha=1.):
    return torch.celu_(inputs, alpha)

# 1. Define a valid test case
valid_test_case = {
    'inputs': torch.randn(3, 4),
    'alpha': 1.5
}

# 2. Parameters affecting output shape: Only 'alpha' affects computation but NOT the shape.
#    The shape is determined solely by 'inputs', which we're excluding.
#    Therefore, no parameters in call_func (except 'inputs') affect output shape.

# 3. Parameter analysis and discretization for 'alpha':
#    Type: scalar (float or int)
#    Continuous parameter affecting numerical computation only
#    Boundary analysis: Typically used in range (0, 2], but can be any real number
#    Discretization: Include boundary values and 5 typical values

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Since no parameters (except 'inputs') affect shape, 
    # we include 'alpha' for completeness as it's the only other parameter
    alpha: list = None
    
    def __post_init__(self):
        if self.alpha is None:
            # Discretized value range for continuous parameter 'alpha'
            # Boundary values: 0 (edge case), 1 (default), 2 (typical)
            # Additional typical values: -1, 0.5, 1.0, 1.5, 2.0
            self.alpha = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]