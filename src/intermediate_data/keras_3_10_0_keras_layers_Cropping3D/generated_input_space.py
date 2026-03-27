import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.random((2, 28, 28, 10, 3)).astype(np.float32),
    "cropping": (2, 4, 2),
    "data_format": None
}

# Task 2: Parameters affecting output shape: cropping and data_format

# Task 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    cropping: List[Union[int, Tuple[int, int, int], 
                        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = field(
        default_factory=lambda: [
            # int examples (symmetric cropping)
            0, 1, 2, 3, 4,
            # tuple of 3 ints examples
            (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4),
            (2, 4, 2),  # From valid_test_case
            # tuple of 3 tuples examples
            ((0, 0), (0, 0), (0, 0)),
            ((1, 1), (1, 1), (1, 1)),
            ((2, 2), (3, 3), (4, 4)),
            ((1, 2), (3, 4), (5, 6)),
            ((0, 1), (0, 1), (0, 1))
        ]
    )
    data_format: List[str] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )