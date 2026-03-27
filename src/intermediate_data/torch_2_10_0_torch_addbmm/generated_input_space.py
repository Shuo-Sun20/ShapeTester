import torch
from dataclasses import dataclass, field
from typing import Optional, Union

# Example tensors from the problem
M = torch.randn(3, 5)
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [M, batch1, batch2],
    'beta': 1,
    'alpha': 1,
    'out': None
}

# Task 2 & 3: Parameter analysis and value space construction
# Only 'out' can affect output shape (by requiring matching shape)
# out: Optional[Tensor] parameter

@dataclass
class InputSpace:
    out: Optional[Union[None, torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.randn(3, 5),
        torch.randn(3, 5),
        torch.randn(3, 5),
        torch.randn(3, 5)
    ])