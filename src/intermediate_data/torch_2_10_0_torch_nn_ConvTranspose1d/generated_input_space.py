import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "in_channels": 16,
    "out_channels": 32,
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "output_padding": 1,
    "groups": 1,
    "bias": True,
    "dilation": 1,
    "inputs": [torch.randn(4, 16, 32)],
    "output_size": None
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that affect output shape (except inputs)
    kernel_size: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    stride: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    output_padding: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    dilation: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    output_size: List[Optional[int]] = field(default_factory=lambda: [None, 10, 20, 30, 40])