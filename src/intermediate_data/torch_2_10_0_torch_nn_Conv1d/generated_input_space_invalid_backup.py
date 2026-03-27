import torch
from dataclasses import dataclass, field

# 1. Valid test case variable
valid_test_case = {
    "in_channels": 16,
    "out_channels": 33,
    "kernel_size": 3,
    "stride": 2,
    "padding": 0,
    "dilation": 1,
    "groups": 1,
    "bias": True,
    "padding_mode": 'zeros',
    "inputs": torch.randn(20, 16, 50)
}

# 2. Parameters affecting output shape: out_channels, kernel_size, stride, padding, dilation
# 3. Value space for each parameter

@dataclass
class InputSpace:
    out_channels: list = field(default_factory=lambda: [1, 8, 16, 32, 64])
    kernel_size: list = field(default_factory=lambda: [1, 3, 5, 7, 9])
    stride: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    dilation: list = field(default_factory=lambda: [1, 2, 3, 4, 5])