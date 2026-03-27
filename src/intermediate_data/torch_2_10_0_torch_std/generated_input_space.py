import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

valid_test_case = {
    "inputs": torch.randn(4, 4),
    "dim": 1,
    "correction": 1,
    "keepdim": True,
    "out": None
}

@dataclass
class InputSpace:
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,
            0, 1, 2, -1, -2,
            (0,), (1,), (0, 1), (0, -1),
            (0, 1, 2), (-1, -2, -3)
        ]
    )
    keepdim: List[bool] = field(default_factory=lambda: [True, False])