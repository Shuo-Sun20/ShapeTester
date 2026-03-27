import torch
from dataclasses import dataclass, field
from typing import Optional, List, Any

valid_test_case = {
    "inputs": [torch.randn(4)],
    "out": None
}

@dataclass
class InputSpace:
    out: List[Optional[Any]] = field(default_factory=lambda: [
        None,
        torch.randn(4),
        torch.randn(4, 3),
        torch.randn(4, 3, 2),
        torch.randn(1, 4)
    ])