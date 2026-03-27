import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

valid_test_case = {
    "inputs": [torch.randn(2, 1, 3, 1, 4)],
    "dim": (1, 3)
}

@dataclass
class InputSpace:
    dim: List[Union[int, Tuple[int, ...], None]] = field(
        default_factory=lambda: [
            None,
            0,
            -1,
            2,
            (1,),
            (0, 2),
            (1, 3),
            (-1, -2),
            (1, 2, 3),
            (0, 1, 2, 3)
        ]
    )