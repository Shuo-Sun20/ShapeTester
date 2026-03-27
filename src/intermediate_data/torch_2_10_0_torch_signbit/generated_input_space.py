import torch
from dataclasses import dataclass, field
from typing import Optional, Union

valid_test_case = {
    'inputs': torch.tensor([0.7, -1.2, 0., 2.3, -0.0, 0.0]),
    'out': None
}

@dataclass
class InputSpace:
    out: Optional[Union[torch.Tensor, type(None)]] = field(
        default_factory=lambda: [
            None,
            torch.tensor([]),
            torch.tensor([False, False]),
            torch.tensor([False, False, False, False, False, False]),
            torch.tensor([[False, False], [False, False]]),
            torch.empty(0, dtype=torch.bool),
        ]
    )