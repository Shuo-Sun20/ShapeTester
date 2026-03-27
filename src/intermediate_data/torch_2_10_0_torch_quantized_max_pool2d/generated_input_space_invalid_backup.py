import torch
from dataclasses import dataclass, field
from typing import List, Union, Tuple

valid_test_case = {
    "inputs": [torch.quantize_per_tensor(torch.rand(2, 2, 2, 2), 1.5, 3, torch.quint8)],
    "kernel_size": [2, 2],
    "stride": [],
    "padding": 0,
    "dilation": 1,
    "ceil_mode": False
}

@dataclass
class InputSpace:
    kernel_size: List[Union[List[int], Tuple[int, int]]] = field(default_factory=lambda: [
        [1, 1], [2, 2], [3, 3], [5, 5], [7, 7]
    ])
    stride: List[Union[List[int], Tuple[int, int]]] = field(default_factory=lambda: [
        [], [1, 1], [2, 2], [1, 2], [2, 1]
    ])
    padding: List[Union[int, List[int], Tuple[int, int]]] = field(default_factory=lambda: [
        0, 1, 2, [1, 2], (2, 1)
    ])
    dilation: List[Union[int, List[int], Tuple[int, int]]] = field(default_factory=lambda: [
        1, 2, 3, [1, 2], (2, 1)
    ])
    ceil_mode: List[bool] = field(default_factory=lambda: [False, True])