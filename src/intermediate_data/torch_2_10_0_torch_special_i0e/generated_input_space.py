import torch
from dataclasses import dataclass, field
from typing import Optional, Union

valid_test_case = {
    'inputs': torch.randn(3, 3, dtype=torch.float32),
    'out': None
}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.zeros(3, 3, dtype=torch.float32),
        torch.ones(3, 3, dtype=torch.float32),
        torch.full((3, 3), 2.5, dtype=torch.float32),
        torch.randn(3, 3, dtype=torch.float32),
        torch.randn(5, 5, dtype=torch.float32),
        torch.randn(1, 10, dtype=torch.float32)
    ])