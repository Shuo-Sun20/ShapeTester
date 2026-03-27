import torch
from dataclasses import dataclass, field
from typing import List, Tuple

def call_func(inputs, shape):
    return torch.broadcast_to(inputs, shape)

# 1. Valid test case
valid_test_case = {
    'inputs': torch.randn(1, 4),
    'shape': (3, 4)
}

# 4. InputSpace class definition
@dataclass
class InputSpace:
    shape: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (1, 4),
        (3, 4),
        (5, 4),
        (2, 3, 4),
        (1, 5, 4)
    ])