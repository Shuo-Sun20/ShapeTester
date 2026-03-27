import torch
from dataclasses import dataclass, field
from typing import Optional

def call_func(inputs, out=None):
    return torch.sign(inputs, out=out)

# 1. Valid test case
example_input = torch.randn(3, 4) * 2 - 1
valid_test_case = {
    'inputs': example_input,
    'out': None
}

# 2. Parameters affecting output shape: only 'inputs' affects shape
#    'out' only affects content/location but not shape

# 3. Parameter value space analysis:
#    'inputs': torch.Tensor - shape determines output shape
#    'out': Optional[torch.Tensor] - can be None or tensor matching input shape

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # Only 'inputs' directly affects output shape
    # Representing tensor shapes as discrete values
    # Using None to represent not providing 'out' parameter
    out: list = field(default_factory=lambda: [None, 
                                              torch.tensor([0.0]), 
                                              torch.randn(3, 4) * 2 - 1,
                                              torch.zeros(3, 4),
                                              torch.ones(3, 4),
                                              torch.full((3, 4), -5.0),
                                              torch.full((3, 4), 5.0)])
    
# Example instantiation
var = InputSpace()