import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

valid_test_case = {
    'inputs': [],
    'size': (2, 3),
    'stride': (3, 1),
    'dtype': torch.float32,
    'layout': None,
    'device': torch.device('cpu'),
    'requires_grad': False,
    'pin_memory': False
}

@dataclass
class InputSpace:
    size: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (0, 0),      # Boundary: empty 2D tensor
        (1, 1),      # Minimal non-empty 2D tensor
        (2, 3),      # Typical small 2D tensor
        (5, 5),      # Typical medium 2D tensor
        (10, 10)     # Boundary: larger 2D tensor
    ])