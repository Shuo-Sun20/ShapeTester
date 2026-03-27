import torch
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    'inputs': [torch.randn(2, 2), torch.randn(2, 2)],
    'out': None
}

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.empty(4, 4),
        torch.zeros(4, 4),
        torch.ones(4, 4),
        torch.randn(4, 4)
    ])