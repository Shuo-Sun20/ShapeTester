import torch
from dataclasses import dataclass

# 1. Define valid_test_case
valid_test_case = {
    "inputs": torch.randn(3, 4)
}

# 2-4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output tensor shape."""
    pass