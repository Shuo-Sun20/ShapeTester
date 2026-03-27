import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# Define a valid test case that can successfully call call_func()
x = keras.random.normal(shape=(3, 4))
valid_test_case = {
    "inputs": [x],
    "ord": None,
    "axis": None,
    "keepdims": False
}

@dataclass
class InputSpace:
    """
    Data class containing all parameters that affect the shape of the output tensor
    from keras.ops.norm, with discretized value ranges.
    """
    axis: List[Optional[Union[int, Tuple[int, int]]]] = field(
        default_factory=lambda: [
            None,
            0,
            1,
            -1,
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, -2)
        ]
    )
    keepdims: List[bool] = field(
        default_factory=lambda: [True, False]
    )