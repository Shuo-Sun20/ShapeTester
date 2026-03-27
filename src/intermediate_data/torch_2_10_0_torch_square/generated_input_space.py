import torch
from dataclasses import dataclass
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(4)],
    "out": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        if self.out is None:
            # Create test tensors with shape (4) to match the valid_test_case
            self.out = [
                None,
                torch.zeros(4),
                torch.ones(4),
                torch.full((4,), 2.0),
                torch.full((4,), -1.0),
                torch.tensor([1.0, 2.0, 3.0, 4.0])
            ]