import torch
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': torch.randn(3, 4),
    'n': torch.tensor(2),
    'out': None
}

@dataclass
class InputSpace:
    n: List[torch.Tensor] = field(default_factory=lambda: [
        torch.tensor(0),
        torch.tensor(1),
        torch.tensor(2),
        torch.tensor(5),
        torch.tensor(10),
        torch.tensor([0, 1, 2]),
        torch.tensor([[0, 1], [2, 3]]),
        torch.tensor([[[0, 1, 2], [3, 4, 5]]]),
        torch.tensor(0.0),
        torch.tensor(1.0),
        torch.tensor(2.0)
    ])