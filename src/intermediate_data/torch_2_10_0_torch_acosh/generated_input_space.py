import torch
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": torch.randn(4).uniform_(1, 2),
    "out": None
}

@dataclass
class InputSpace:
    # 'out' is the only parameter besides 'inputs' that can affect output shape
    # It affects shape because if provided, output will have same shape as 'out'
    out: list = field(default_factory=lambda: [
        None,
        torch.zeros(1, dtype=torch.float32),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(2, 2, dtype=torch.float32),
        torch.zeros(1, 3, 1, dtype=torch.float32)
    ])