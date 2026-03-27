import torch
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, offset=0, dim1=0, dim2=1):
    input_tensor, src_tensor = inputs
    return torch.diagonal_scatter(input_tensor, src_tensor, offset, dim1, dim2)

# Task 1: valid_test_case variable
input_tensor = torch.randn(4, 4)
src_tensor = torch.diagonal(input_tensor, offset=0, dim1=0, dim2=1)
valid_test_case = {
    'inputs': [input_tensor, src_tensor],
    'offset': 0,
    'dim1': 0,
    'dim2': 1
}

# Task 4: InputSpace class
@dataclass
class InputSpace:
    offset: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    dim1: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    dim2: List[int] = field(default_factory=lambda: [0, 1, 2, 3])