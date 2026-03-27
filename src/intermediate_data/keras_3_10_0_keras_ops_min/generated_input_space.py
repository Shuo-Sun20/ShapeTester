import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

def call_func(inputs, axis=None, keepdims=False, initial=None):
    return keras.ops.min(x=inputs, axis=axis, keepdims=keepdims, initial=initial)

# 1. Define valid test case
valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4)),
    "axis": None,
    "keepdims": False,
    "initial": None
}

# 2. Parameters affecting output shape: axis, keepdims
# 3. Discretized value spaces for parameters affecting output shape

@dataclass
class InputSpace:
    # For axis parameter: None, int, tuple of ints
    axis: List[Union[None, int, Tuple[int, ...]]] = field(
        default_factory=lambda: [None, 0, 1, -1, -2, (0,), (1,), (0, 1), (-1, -2), (0, -1)]
    )
    
    # For keepdims parameter: boolean
    keepdims: List[bool] = field(
        default_factory=lambda: [True, False]
    )