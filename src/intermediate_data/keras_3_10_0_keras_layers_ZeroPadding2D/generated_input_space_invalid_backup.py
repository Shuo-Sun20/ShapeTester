import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case
valid_test_case = {
    "padding": ((2, 2), (1, 1)),
    "data_format": "channels_last",
    "inputs": np.random.randn(2, 32, 32, 3).astype(np.float32)
}

# 2. Parameters affecting output shape (except "inputs"): padding, data_format

@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]] = field(
        default_factory=lambda: [
            # int: symmetric padding
            0,  # min value
            1,  # typical value
            2,  # typical value
            3,  # typical value
            5   # boundary/typical value
        ]
    )
    data_format: List[str] = field(
        default_factory=lambda: [
            "channels_last",
            "channels_first"
        ]
    )