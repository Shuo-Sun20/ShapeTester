import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': [torch.randn(100, 128), torch.randn(100, 128)],
    'p': 2,
    'eps': 1e-6,
    'keepdim': False
}

@dataclass
class InputSpace:
    keepdim: List[bool] = field(default_factory=lambda: [False, True])