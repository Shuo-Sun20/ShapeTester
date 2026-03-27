import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Union, Optional

def call_func(inputs, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    input_tensor = inputs[0]
    return F.avg_pool1d(input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad)

# 1. Valid test case dictionary
valid_test_case = {
    "inputs": [torch.randn(2, 3, 10)],
    "kernel_size": 3,
    "stride": 2,
    "padding": 0,
    "ceil_mode": False,
    "count_include_pad": True
}

# 2. Parameters affecting output shape (excluding inputs):
#    - kernel_size, stride, padding, ceil_mode

@dataclass
class InputSpace:
    kernel_size: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 3, 5, 7, (3, 1)])
    stride: List[Optional[int]] = field(default_factory=lambda: [None, 1, 2, 3, 5])
    padding: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    ceil_mode: List[bool] = field(default_factory=lambda: [True, False])