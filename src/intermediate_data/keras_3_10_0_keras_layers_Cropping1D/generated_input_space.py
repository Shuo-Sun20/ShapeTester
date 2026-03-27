import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Valid test case
valid_test_case = {
    "cropping": 2,
    "inputs": np.random.randn(4, 10, 5)
}

# 4. InputSpace class definition
@dataclass
class InputSpace:
    cropping: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        # Integer values (0-4)
        0, 1, 2, 3, 4,
        # Tuple values (left, right)
        (0, 0), (1, 0), (0, 1), (1, 1), (2, 2),
        (3, 0), (0, 3), (2, 3), (3, 2), (4, 4)
    ])