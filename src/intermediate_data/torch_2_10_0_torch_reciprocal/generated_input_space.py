import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. valid_test_case definition
valid_test_case = {
    "inputs": [torch.randn(4)],
    "out": None
}

# 4. InputSpace class definition
@dataclass
class InputSpace:
    out: List[Optional[Union[torch.Tensor, None]]] = field(default_factory=lambda: [
        None,
        torch.randn(4),
        torch.ones(4),
        torch.zeros(4),
        torch.full((4,), 0.5)
    ])