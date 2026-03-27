import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(5),
    "out": None
}

# 2. Identify parameters affecting output shape (excluding "inputs")
# Only "out" parameter can affect output shape if provided

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters (except "inputs") that can affect 
    the shape of the output tensor.
    """
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,  # Default case - output shape matches input
            torch.zeros(3, 2),  # Example of valid output tensor
            torch.zeros(1, 1, 1),  # Single element tensor
            torch.zeros(5),  # 1D tensor
            torch.zeros(2, 3, 4),  # 3D tensor
            torch.zeros(0),  # Empty tensor
            torch.zeros(0, 2, 0),  # Empty tensor with multiple dimensions
        ]
    )