import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

valid_test_case = {
    "inputs": torch.randn(4, 4),
    "dim": 1,
    "keepdim": False,
    "out": None
}

@dataclass
class InputSpace:
    dim: List[Union[None, int, Tuple[int, ...]]] = field(
        default_factory=lambda: [None, 0, 1, (0, 1), (1, 0)]
    )
    keepdim: List[bool] = field(default_factory=lambda: [True, False])