import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, List

def call_func(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inputs=None):
    conv3d_layer = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    output = conv3d_layer(inputs)
    return output

# 1. Define valid_test_case
valid_test_case = {
    'in_channels': 3,
    'out_channels': 6,
    'kernel_size': (3, 5, 5),
    'stride': (2, 1, 1),
    'padding': (1, 2, 2),
    'dilation': 1,
    'groups': 1,
    'bias': True,
    'padding_mode': 'zeros',
    'inputs': torch.randn(2, 3, 16, 32, 32)
}

@dataclass
class InputSpace:
    """Dataclass containing parameters that affect output shape with discretized value spaces"""
    out_channels: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])
    kernel_size: List[Union[int, Tuple[int, int, int]]] = field(default_factory=lambda: [
        1, 3, (3, 5, 5), 5, (7, 7, 7)
    ])
    stride: List[Union[int, Tuple[int, int, int]]] = field(default_factory=lambda: [
        1, 2, (1, 1, 1), (2, 1, 1), (2, 2, 2)
    ])
    padding: List[Union[int, str, Tuple[int, int, int]]] = field(default_factory=lambda: [
        0, 1, (1, 2, 2), 'valid', 'same'
    ])
    dilation: List[Union[int, Tuple[int, int, int]]] = field(default_factory=lambda: [
        1, 2, (1, 1, 1), (2, 2, 2), (1, 2, 3)
    ])