import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

valid_test_case = {
    'inputs': [torch.randn(20, 16, 50), torch.randn(16, 33, 5)],
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'groups': 1,
    'dilation': 1
}

@dataclass
class InputSpace:
    stride: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [1, 2, 3, (1,), (2,)])
    padding: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [0, 1, 2, (0,), (1,)])
    output_padding: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [0, 1, 2, (0,), (1,)])
    groups: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    dilation: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [1, 2, 3, (1,), (2,)])