import torch
from dataclasses import dataclass, field
from typing import Optional

valid_test_case = {
    "inputs": [torch.randn(5), torch.randn(5)],
    "out": None
}

@dataclass
class InputSpace:
    out: list = field(default_factory=lambda: [
        None,
        torch.tensor(0.0),
        torch.tensor([0.0]),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.0], device='cpu'),
        torch.tensor([0.0], device='cuda') if torch.cuda.is_available() else torch.tensor([0.0], device='cpu')
    ])