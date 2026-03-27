import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
torch.manual_seed(42)
a = torch.randn(2, 3)
b = torch.randn(2, 4, 3)
valid_test_case = {
    'inputs': [a, b],
    'out': None
}

# Task 2,3,4: Define InputSpace class
@dataclass
class InputSpace:
    # The 'out' parameter is the only parameter besides 'inputs' that can be provided to call_func
    # It doesn't directly affect the output shape, but must match the correct output shape
    # We create 5 possible values for 'out' parameter
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(2, 2, 4),  # Correct shape for example inputs
        torch.zeros(2, 2, 4),   # Another correct shape option
        torch.ones(2, 2, 4),    # Another correct shape option
        torch.randn(2, 2, 4)    # Another correct shape option
    ])