import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.random((2, 28, 28, 10, 3)).astype(np.float32),
    "cropping": (2, 4, 2),
    "data_format": None
}

# Task 2: Parameters affecting output shape: cropping, data_format

# Task 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # cropping: int OR tuple of 3 ints OR tuple of 3 tuples of 2 ints
    cropping: List[Union[int, Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = field(
        default_factory=lambda: [
            # int values (symmetric cropping)
            0, 2, 5,
            # tuple of 3 ints (symmetric per dimension)
            (1, 2, 3),
            # tuple of 3 tuples (asymmetric cropping)
            ((0, 1), (2, 2), (1, 3))
        ]
    )
    
    # data_format: None OR "channels_last" OR "channels_first"
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )