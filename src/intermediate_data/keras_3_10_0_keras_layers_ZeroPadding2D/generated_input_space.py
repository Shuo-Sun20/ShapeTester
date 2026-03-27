import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    "padding": ((2, 2), (1, 1)),
    "data_format": "channels_last",
    "inputs": np.random.randn(2, 32, 32, 3).astype(np.float32)
}

# 2. Identify parameters affecting output shape (except "inputs")
# padding and data_format affect output shape

# 3. Discretize parameter value spaces

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]] = field(
        default_factory=lambda: [
            # int type (symmetric padding)
            0, 1, 2, 3, 4,
            # tuple of 2 ints (symmetric_height_pad, symmetric_width_pad)
            (0, 0), (1, 1), (1, 2), (2, 1), (3, 4),
            # tuple of 2 tuples of 2 ints ((top, bottom), (left, right))
            ((0, 0), (0, 0)), ((1, 1), (1, 1)), ((1, 2), (3, 4)),
            ((2, 2), (1, 1)),  # From valid_test_case
            ((0, 1), (2, 0))
        ]
    )
    data_format: List[str] = field(
        default_factory=lambda: [
            "channels_last",
            "channels_first"
        ]
    )