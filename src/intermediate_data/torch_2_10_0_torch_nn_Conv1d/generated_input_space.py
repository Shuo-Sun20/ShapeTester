from dataclasses import dataclass, field
import torch

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

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    out_channels: list = field(default_factory=lambda: [16, 32, 33, 64, 128, 256])
    kernel_size: list = field(default_factory=lambda: [1, 3, 5, 7, 11])
    stride: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: list = field(default_factory=lambda: [0, 1, 2, 3, 5, 'valid', 'same'])
    dilation: list = field(default_factory=lambda: [1, 2, 3, 4, 5])