import torch
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': [torch.randn(3, 3)],
    'offset': 0,
    'dim1': 0,
    'dim2': 1
}

@dataclass
class InputSpace:
    offset: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    dim1: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    dim2: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])