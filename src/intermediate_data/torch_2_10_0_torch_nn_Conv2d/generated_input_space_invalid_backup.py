import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    'in_channels': 16,
    'out_channels': 33,
    'kernel_size': 3,
    'stride': 2,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
    'bias': True,
    'padding_mode': 'zeros',
    'inputs': torch.randn(20, 16, 50, 100)
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    kernel_size: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,
        3,
        5,
        7,
        (3, 5)
    ])
    stride: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,
        2,
        3,
        (1, 2),
        (2, 1)
    ])
    padding: List[Union[int, Tuple[int, int], str]] = field(default_factory=lambda: [
        0,
        1,
        2,
        (1, 2),
        'same'
    ])
    dilation: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,
        2,
        3,
        (1, 2),
        (2, 1)
    ])
    out_channels: List[int] = field(default_factory=lambda: [
        16,
        32,
        64,
        128,
        256
    ])