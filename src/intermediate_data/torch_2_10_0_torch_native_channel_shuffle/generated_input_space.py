import torch
from typing import List
from dataclasses import dataclass, field

valid_test_case = {
    'inputs': [torch.randn(2, 8, 4, 4)],
    'groups': 2
}

@dataclass
class InputSpace:
    groups: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])