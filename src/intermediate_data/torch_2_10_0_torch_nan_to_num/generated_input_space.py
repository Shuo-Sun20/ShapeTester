import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
torch.manual_seed(42)
x = torch.randn(3, 3)
x[0, 0] = float('nan')
x[1, 1] = float('inf')
x[2, 2] = -float('inf')

valid_test_case = {
    'inputs': [x],
    'nan': 0.0,
    'posinf': None,
    'neginf': None,
    'out': None
}

# Task 4: Define InputSpace with discretized value ranges
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros(3, 3, dtype=torch.float32),
        torch.ones(3, 3, dtype=torch.float32),
        torch.full((3, 3), 2.0, dtype=torch.float32),
        torch.randn(3, 3)
    ])