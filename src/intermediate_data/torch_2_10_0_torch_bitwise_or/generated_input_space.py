import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Valid test case
valid_test_case = {
    'inputs': [
        torch.tensor([-1, -2, 3], dtype=torch.int8),
        torch.tensor([1, 0, 3], dtype=torch.int8)
    ],
    'out': None
}

# 2/3/4. InputSpace dataclass
@dataclass
class InputSpace:
    # Only 'out' parameter affects output shape besides 'inputs'
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # No pre-allocated output tensor
        torch.empty((3,), dtype=torch.int8),  # Exact shape match
        torch.empty((1, 3), dtype=torch.int8),  # Broadcastable shape
        torch.empty((3, 1), dtype=torch.int8),  # Broadcastable shape
        torch.empty((1, 1, 3), dtype=torch.int8)  # Broadcastable shape
    ])