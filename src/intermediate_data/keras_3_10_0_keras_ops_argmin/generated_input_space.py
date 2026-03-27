import keras
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union

# 1. valid_test_case definition
valid_test_case = {
    'inputs': keras.random.normal(shape=(3, 4, 5)),
    'axis': 1,
    'keepdims': False
}

# 2. Parameters affecting output shape (except "inputs"): axis, keepdims

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # axis can be None, integer, or list of integers
    axis: List[Optional[Union[int, List[int]]]] = None
    keepdims: List[bool] = None
    
    def __post_init__(self):
        if self.axis is None:
            # For a 3D tensor (3,4,5) - common case from example
            # Includes: None (flatten), single axis (0,1,2), negative axis (-1,-2,-3), 
            # and multi-axis cases (boundary and typical)
            self.axis = [
                None,                     # flatten entire tensor
                0, 1, 2,                  # positive axes
                -1, -2, -3,               # negative axes (boundary values)
                (0, 1), (0, 2), (1, 2),   # 2D combinations
                (0, 1, 2),                # all axes
                [0, 1], [0, 2], [1, 2]    # list versions
            ]
        if self.keepdims is None:
            self.keepdims = [False, True]  # only two discrete values

# Example instantiation
var = InputSpace()