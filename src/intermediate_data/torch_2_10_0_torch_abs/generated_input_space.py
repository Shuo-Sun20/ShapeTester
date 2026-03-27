import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(3, 4) * 5],
    "out": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Task 2: Only 'out' parameter affects output shape besides 'inputs'
    # Task 3: Discretized value space for 'out' (5 values max)
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.tensor([]),
            torch.tensor([1.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ]
    )