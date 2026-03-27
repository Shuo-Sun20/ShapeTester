import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

valid_test_case = {
    "kernel_size": (2, 3),
    "dilation": 1,
    "padding": 0,
    "stride": 1,
    "inputs": [torch.randn(2, 3, 4, 5)]
}

@dataclass
class InputSpace:
    kernel_size: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5, (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    dilation: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5, (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])
    padding: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [0, 1, 2, 3, 4, (0, 0), (0, 1), (1, 0), (1, 1), (2, 2)])
    stride: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5, (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)])