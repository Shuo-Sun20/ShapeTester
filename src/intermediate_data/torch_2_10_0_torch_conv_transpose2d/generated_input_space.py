import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

valid_test_case = {
    'inputs': [torch.randn(1, 4, 5, 5), torch.randn(4, 8, 3, 3)],
    'bias': None,
    'stride': 1,
    'padding': 1,
    'output_padding': 0,
    'groups': 1,
    'dilation': 1
}

@dataclass
class InputSpace:
    """Parameters that affect the output shape of torch.conv_transpose2d"""
    stride: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2), (2, 1)]
    )
    padding: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [0, 1, 2, (0, 1), (1, 0)]
    )
    output_padding: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [0, 1, 2, (0, 1), (1, 0)]
    )
    dilation: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2), (2, 1)]
    )
    groups: List[int] = field(default_factory=lambda: [1, 2, 4])