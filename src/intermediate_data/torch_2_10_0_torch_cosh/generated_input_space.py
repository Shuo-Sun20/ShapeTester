import torch
from dataclasses import dataclass
from typing import Optional

# Set seed for reproducibility
torch.manual_seed(42)
inputs = torch.randn(4)
valid_test_case = {"inputs": inputs, "out": None}

@dataclass
class InputSpace:
    out: list[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        if self.out is None:
            # Create sample tensors of different shapes for testing
            # Using the same shape as valid_test_case['inputs'] for consistency
            base_tensor = valid_test_case['inputs']
            self.out = [
                None,
                torch.empty_like(base_tensor),  # Same shape as input
                torch.empty(1, 4),             # Row vector
                torch.empty(4, 1),             # Column vector
                torch.empty(2, 2)              # 2x2 matrix
            ]