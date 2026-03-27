import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

valid_test_case = {
    'inputs': [torch.randn(4)],
    'out': None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(
        default_factory=lambda: [
            None,
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(3, 4, dtype=torch.float32),
            torch.zeros(1, 2, 3, dtype=torch.float64),
            torch.zeros(2, 3, 4, 5, dtype=torch.float32)
        ]
    )