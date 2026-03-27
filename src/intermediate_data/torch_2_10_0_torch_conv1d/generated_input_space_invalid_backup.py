import torch
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(2, 3, 10)],
    "weight": torch.randn(4, 3, 3),
    "bias": None,
    "stride": 1,
    "padding": 0,
    "dilation": 1,
    "groups": 1
}

# 2-4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # These are all parameters from call_func (except inputs) that affect output shape
    weight: List[torch.Tensor] = field(default_factory=lambda: [
        torch.randn(4, 3, 3),   # Standard case
        torch.randn(8, 3, 5),   # Larger kernel and more channels
        torch.randn(2, 3, 1),   # Kernel size 1
        torch.randn(1, 3, 7),   # Single output channel
        torch.randn(4, 3, 10),  # Kernel size equals input length
    ])
    stride: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: List[Union[int, str]] = field(default_factory=lambda: [0, 1, 2, 'same', 'valid'])
    dilation: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    groups: List[int] = field(default_factory=lambda: [1, 2, 3])