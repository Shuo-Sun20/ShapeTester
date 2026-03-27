import torch
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': None,
    'window_length': 10,
    'periodic': True,
    'dtype': torch.float32,
    'layout': torch.strided,
    'device': torch.device('cpu'),
    'requires_grad': False
}

# Task 4: Define InputSpace dataclass with discretized value ranges for parameters affecting output shape
@dataclass
class InputSpace:
    window_length: list = field(default_factory=lambda: [1, 2, 5, 10, 20])
    periodic: list = field(default_factory=lambda: [True, False])