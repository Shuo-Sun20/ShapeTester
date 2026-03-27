import torch
from dataclasses import dataclass, field
from typing import Union, List

# Generate random tensors for testing
torch.manual_seed(42)
base_tensor = torch.randn(3, 2)
exp_tensor = torch.randint(1, 4, (3, 2)).float()
scalar_base = 2.5

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': base_tensor,
    'exponent': 2
}

# Task 4: Define InputSpace class
@dataclass
class InputSpace:
    exponent: List[Union[int, float, torch.Tensor]] = field(default_factory=lambda: [
        0,
        1,
        2,
        0.5,
        torch.tensor(3.0)
    ])