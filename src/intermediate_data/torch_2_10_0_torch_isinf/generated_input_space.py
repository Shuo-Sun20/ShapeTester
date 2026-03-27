import torch
from dataclasses import dataclass, field
from typing import List, Union

def call_func(inputs):
    return torch.isinf(inputs)

# Generate random tensor with some infinite values
input_tensor = torch.randn(5)
input_tensor[1] = float('inf')
input_tensor[3] = float('-inf')

# 1. Valid test case
valid_test_case = {'inputs': input_tensor}

# 2. & 3. & 4. InputSpace class
@dataclass
class InputSpace:
    inputs: List[Union[torch.Tensor, float, int]] = field(default_factory=lambda: [
        torch.tensor(1.0),  # scalar
        torch.tensor(float('inf')),  # scalar infinite
        torch.tensor([1.0, float('inf'), 2.0, float('-inf'), 3.0]),  # 1D mixed
        torch.tensor([[1.0, 2.0], [3.0, float('inf')]]),  # 2D
        torch.tensor([[[1.0], [2.0]], [[float('inf')], [4.0]]])  # 3D
    ])