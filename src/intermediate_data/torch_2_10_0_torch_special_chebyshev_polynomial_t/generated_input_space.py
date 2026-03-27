import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": (torch.randn(3, 4), torch.tensor(3, dtype=torch.int32)),
    "out": None
}

# Task 2 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameter affecting output shape (except "inputs"):
    # Only "out" parameter can affect output shape through pre-allocation
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # No pre-allocation
        torch.empty(3, 4),  # Same shape as input
        torch.empty(3, 4, dtype=torch.float64),  # Different dtype
        torch.empty(6, 8),  # Larger shape
        torch.empty(1, 4),  # Smaller shape (broadcastable)
        torch.empty(3, 4, device='cpu'),  # Explicit device
    ])