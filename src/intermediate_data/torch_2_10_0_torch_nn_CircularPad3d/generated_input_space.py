import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

valid_test_case = {
    "padding": 3,
    "inputs": torch.randn(16, 3, 8, 320, 480)
}

@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int, int, int, int, int]]] = field(
        default_factory=lambda: [
            # Integer padding values
            -8, -5, -2, 0, 2, 3, 5, 8,
            # Tuple padding values (format: (left, right, top, bottom, front, back))
            (0, 0, 0, 0, 0, 0),
            (3, 3, 6, 6, 1, 1),
            (-1, -1, -1, -1, -1, -1),
            (2, 2, 2, 2, 2, 2),
            (5, 5, 10, 10, 3, 3),
            (0, 5, 0, 10, 0, 3),
            (-2, 2, -5, 5, -1, 1),
            (8, -4, 320, -100, 480, -200)
        ]
    )