import torch
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Define a valid test case
input_tensor = torch.tensor([-1.5, 0, 2.0])
values_tensor = torch.tensor([0.5])
valid_test_case = {'inputs': [input_tensor, values_tensor], 'out': None}

# Tasks 2, 3, and 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains parameters that affect the shape of the output tensor.
    For the torch.heaviside API, only the 'out' parameter can affect the shape.
    """
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([-1.0, 0.0, 1.0])
    ])