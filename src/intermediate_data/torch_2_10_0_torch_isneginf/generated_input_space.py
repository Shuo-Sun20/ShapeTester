import torch
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    'inputs': torch.tensor([-float('inf'), float('inf'), torch.randn(1).item()]),
    'out': None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([], dtype=torch.bool),
        torch.tensor([True], dtype=torch.bool),
        torch.tensor([True, False], dtype=torch.bool),
        torch.tensor([True, False, True], dtype=torch.bool)
    ])