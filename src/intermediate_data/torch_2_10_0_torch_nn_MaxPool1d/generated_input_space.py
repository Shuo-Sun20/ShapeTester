import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, List

valid_test_case = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 0,
    "dilation": 1,
    "return_indices": False,
    "ceil_mode": False,
    "inputs": torch.randn(20, 16, 50)
}

@dataclass
class InputSpace:
    kernel_size: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7, 10, 15])
    stride: List[Union[int, None]] = field(default_factory=lambda: [1, 2, 3, 4, 5, None])
    padding: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    dilation: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    ceil_mode: List[bool] = field(default_factory=lambda: [True, False])