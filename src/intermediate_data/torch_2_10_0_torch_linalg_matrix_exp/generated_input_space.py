import torch
from dataclasses import dataclass, field
from typing import List

def call_func(inputs):
    return torch.linalg.matrix_exp(inputs[0])

# 1. valid_test_case definition
valid_test_case = {
    "inputs": [torch.randn(3, 3)]
}

# 2-4. InputSpace class definition
@dataclass
class InputSpace:
    """
    Dataclass containing parameters affecting the shape of torch.linalg.matrix_exp output.
    Only tensor shape affects output shape through batch dimensions and matrix size.
    Since call_func only has 'inputs' parameter, there are no additional shape-affecting parameters.
    """
    # No fields needed since call_func only has 'inputs' parameter
    pass