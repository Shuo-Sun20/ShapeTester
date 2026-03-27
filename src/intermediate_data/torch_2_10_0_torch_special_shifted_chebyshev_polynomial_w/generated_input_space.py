import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(3, 4), torch.tensor(2)],
    'out': None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only parameters from call_func that affect output shape
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.randn(3, 4),
        torch.zeros(3, 4),
        torch.ones(3, 4),
        torch.empty(3, 4)
    ])