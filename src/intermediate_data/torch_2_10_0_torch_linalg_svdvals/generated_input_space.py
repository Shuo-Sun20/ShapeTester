import torch
from dataclasses import dataclass, field
from typing import Optional

valid_test_case = {
    'inputs': torch.randn(5, 3),
    'driver': None,
    'out': None
}

@dataclass
class InputSpace:
    out: Optional[list] = field(default_factory=lambda: [
        None,
        torch.zeros(3),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.complex64),
        torch.zeros(3, dtype=torch.complex128),
        torch.tensor([0.0, 0.0, 0.0], device='cpu')
    ])