import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

# 1. Define valid_test_case
valid_test_case = {
    "pool_size": 3,
    "strides": None,
    "padding": "valid",
    "data_format": None,
    "name": None,
    "inputs": np.random.rand(2, 30, 30, 30, 3).astype("float32")
}

# 4. Define InputSpace class
@dataclass
class InputSpace:
    pool_size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (2, 3, 4), (4, 4, 4)]
    )
    strides: List[Optional[Union[int, Tuple[int, int, int]]]] = field(
        default_factory=lambda: [None, 1, 2, (1, 2, 1), (2, 2, 2)]
    )
    padding: List[str] = field(
        default_factory=lambda: ["valid", "same"]
    )
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )