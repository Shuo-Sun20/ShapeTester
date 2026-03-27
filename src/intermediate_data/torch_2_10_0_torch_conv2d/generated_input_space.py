import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# Task 1: Define valid_test_case
batch_size = 2
in_channels = 4
out_channels = 6
input_height = 5
input_width = 5
kernel_height = 3
kernel_width = 3

input_tensor = torch.randn(batch_size, in_channels, input_height, input_width)
weight_tensor = torch.randn(out_channels, in_channels, kernel_height, kernel_width)
inputs = [input_tensor, weight_tensor]

valid_test_case = {
    'inputs': inputs,
    'bias': None,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
    'groups': 1
}

# Task 2 & 3 & 4: Define InputSpace
@dataclass
class InputSpace:
    stride: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 2, 3, (1,2), (2,3)])
    padding: List[Union[int, str, Tuple[int, int]]] = field(default_factory=lambda: [0, 1, 2, (1,1), 'same'])
    dilation: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 2, 3, (1,2), (2,3)])
    groups: List[int] = field(default_factory=lambda: [1, 2])