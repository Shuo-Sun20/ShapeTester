import torch
from dataclasses import dataclass, field

# 1. Define valid_test_case
input_tensor = torch.randn(3)
valid_test_case = {
    "inputs": [input_tensor],
    "out": None
}

# 3. & 4. Define InputSpace dataclass for parameters affecting output shape
@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.randn(3),           # Same shape as input
        torch.randn(1, 3),        # Compatible shape (broadcastable)
        torch.randn(2, 3),        # Compatible shape (broadcastable)
        torch.randn(3, 1),        # Compatible shape (broadcastable)
        torch.randn(3, 3),        # Same shape but different dimensions
    ])