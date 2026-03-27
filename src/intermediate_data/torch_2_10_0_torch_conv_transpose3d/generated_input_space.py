import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. valid_test_case definition
valid_test_case = {
    "inputs": [
        torch.randn(20, 16, 50, 10, 20),  # input_tensor
        torch.randn(16, 33, 3, 3, 3),     # weight
        torch.randn(33)                    # bias
    ],
    "stride": 1,
    "padding": 0,
    "output_padding": 0,
    "groups": 1,
    "dilation": 1
}

# 2 & 3 & 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    stride: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 2, (1, 2, 3), (2, 2, 2), (3, 3, 3)]
    )
    padding: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [0, 1, 2, (0, 1, 2), (1, 1, 1)]
    )
    output_padding: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [0, 1, 2, (0, 1, 2), (1, 1, 1)]
    )
    groups: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16]
    )
    dilation: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2, 3), (2, 2, 2)]
    )