import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
valid_test_case = {
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'ceil_mode': False,
    'count_include_pad': False,
    'inputs': torch.randn(2, 3, 10)
}

# 2 & 3 & 4. Define InputSpace class
@dataclass
class InputSpace:
    # Parameters that affect output shape:
    kernel_size: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7, 10])
    stride: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    padding: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    ceil_mode: List[bool] = field(default_factory=lambda: [True, False])