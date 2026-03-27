import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Generate random input tensors as in the example
A = torch.randn(3, 3).triu_()  # Upper triangular matrix
B = torch.randn(3, 2)           # Right-hand side matrix
inputs = [A, B]

valid_test_case = {
    'inputs': inputs,
    'upper': True,
    'left': True,
    'unitriangular': False,
    'out': None
}

@dataclass
class InputSpace:
    left: List[bool] = field(default_factory=lambda: [True, False])