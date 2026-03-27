import torch
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(4)],
    'n': None,
    'dim': -1,
    'norm': None,
    'out': None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    n: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    dim: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])