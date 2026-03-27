import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional

# 1. Valid test case
valid_test_case = {
    'in_channels': 16,
    'out_channels': 33,
    'kernel_size': 3,
    'inputs': torch.randn(20, 16, 50, 100),
    'stride': 2,
    'padding': 0,
    'output_padding': 0,
    'groups': 1,
    'bias': True,
    'dilation': 1,
    'output_size': None
}

# 2 & 3 & 4. InputSpace definition
@dataclass
class InputSpace:
    # Parameters that affect output shape
    in_channels: list[int] = field(default_factory=lambda: [1, 3, 16, 64])
    out_channels: list[int] = field(default_factory=lambda: [1, 16, 32, 64, 128])
    kernel_size: list[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 3, 5, (3, 5)])
    stride: list[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 2, (1, 2), (2, 1)])
    padding: list[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [0, 1, 2, (0, 1), (1, 2)])
    output_padding: list[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [0, 1, 2, (0, 1)])
    dilation: list[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 2, (1, 2), (2, 1)])
    groups: list[int] = field(default_factory=lambda: [1, 2, 4])
    output_size: list[Optional[Tuple[int, int]]] = field(default_factory=lambda: [None, (50, 100), (100, 200)])