from dataclasses import dataclass, field
from typing import Union, List
import torch

def call_func(inputs, n, out=None):
    input_tensor = inputs
    return torch.special.chebyshev_polynomial_u(input_tensor, n, out=out)

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(4, 3),
    'n': 2,
    'out': None
}

# Task 2-4: Define InputSpace class with discretized value spaces
@dataclass
class InputSpace:
    # Parameter 'inputs' affects output shape (omitted per task 2 requirement)
    # Parameter 'n' affects output shape through broadcasting when it's a tensor
    n: List[Union[int, torch.Tensor]] = field(default_factory=lambda: [
        # Integer values (discrete parameter)
        0, 1, 2, 3, 5, 6, 10, 20,  # Typical values including boundaries and example value (2)
        # Tensor values (broadcasting scenarios)
        torch.tensor(0),                      # Scalar
        torch.tensor([0, 1, 2]),               # 1D tensor
        torch.tensor([[0], [1], [2]]),         # 2D column tensor
        torch.tensor([[0, 1, 2]]),             # 2D row tensor
        torch.tensor([[[0, 1, 2]]]),           # 3D tensor
        torch.tensor([0, 1, 2], dtype=torch.float32),  # Different dtype
        torch.tensor([0, 1, 2], device='cpu'), # Explicit device
    ])