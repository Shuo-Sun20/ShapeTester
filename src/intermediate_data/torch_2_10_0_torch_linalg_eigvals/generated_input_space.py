import torch
from dataclasses import dataclass
from typing import Optional

torch.manual_seed(42)
valid_test_case = {
    "inputs": torch.randn(2, 2, dtype=torch.float64),
    "out": None
}

@dataclass
class InputSpace:
    out: Optional[list] = None
    
    def __post_init__(self):
        # Since 'out' doesn't have value spaces that affect shape, we set it to None
        # The actual value space for 'out' would be tensors matching the output shape,
        # but this doesn't affect the output shape itself
        self.out = [None]