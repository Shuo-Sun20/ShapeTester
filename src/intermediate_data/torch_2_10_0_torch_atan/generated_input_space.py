import torch
from dataclasses import dataclass
from typing import Optional

valid_test_case = {
    'inputs': torch.randn(4),
    'out': None
}

@dataclass
class InputSpace:
    # Only "inputs" parameter affects output shape
    # Parameter "out" does not affect output shape - it must match input shape if provided
    pass  # Empty class since no other parameters affect shape