from dataclasses import dataclass, field
from typing import Union, List
import torch

valid_test_case = {
    "padding": (2, 3),
    "inputs": torch.randn(2, 3, 10)
}

@dataclass
class InputSpace:
    padding: List[Union[int, tuple]] = field(default_factory=lambda: [
        # Int values
        0, 1, 2, 5, 10,
        # Tuple values (padding_left, padding_right)
        (0, 0), (1, 1), (2, 3), (5, 0), (0, 5), (10, 10)
    ])