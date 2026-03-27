import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [],
    'window_length': 10,
    'periodic': True,
    'alpha': 0.54,
    'beta': 0.46,
    'dtype': torch.float32,
    'layout': None,
    'device': 'cpu',
    'pin_memory': False,
    'requires_grad': False
}

# 2 & 3 & 4. Define InputSpace class
@dataclass
class InputSpace:
    # Only window_length affects the output tensor shape
    window_length: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20])
    # Other parameters don't affect shape but included for completeness
    periodic: List[Optional[bool]] = field(default_factory=lambda: [None, True, False])
    alpha: List[Optional[float]] = field(default_factory=lambda: [None, 0.54, 0.5, 0.6, 0.7])
    beta: List[Optional[float]] = field(default_factory=lambda: [None, 0.46, 0.5, 0.4, 0.3])
    dtype: List[Optional[torch.dtype]] = field(default_factory=lambda: [None, torch.float32, torch.float64])
    device: List[Optional[str]] = field(default_factory=lambda: [None, 'cpu'])
    pin_memory: List[bool] = field(default_factory=lambda: [False, True])
    requires_grad: List[bool] = field(default_factory=lambda: [False, True])