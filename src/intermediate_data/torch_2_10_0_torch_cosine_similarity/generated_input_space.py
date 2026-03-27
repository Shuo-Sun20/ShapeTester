import torch
from dataclasses import dataclass, field
from typing import List

# Define the function call_func as in the example
def call_func(inputs, dim=1, eps=1e-8):
    x1, x2 = inputs
    return torch.cosine_similarity(x1, x2, dim=dim, eps=eps)

# Generate random input tensors as in the example
torch.manual_seed(42)
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [input1, input2],
    'dim': 1,
    'eps': 1e-8
}

# Task 4: Define InputSpace as a dataclass
@dataclass
class InputSpace:
    # Parameter that affects output shape (excluding 'inputs'): dim
    # Value space for dim: discrete integer values valid for 2D tensors (100, 128)
    dim: List[int] = field(default_factory=lambda: [-2, -1, 0, 1])