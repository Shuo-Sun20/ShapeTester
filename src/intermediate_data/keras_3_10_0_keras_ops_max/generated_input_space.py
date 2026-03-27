import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple

def call_func(inputs, axis=None, keepdims=False, initial=None):
    return keras.ops.max(inputs, axis=axis, keepdims=keepdims, initial=initial)

# Task 1: Define valid_test_case
example_input = keras.random.uniform((3, 4, 5), minval=-5, maxval=5)
valid_test_case = {
    "inputs": example_input,
    "axis": 1,
    "keepdims": True,
    "initial": None
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    axis: List[Union[int, Tuple[int, ...], None]] = field(default_factory=lambda: [
        # Scalar axis values (positive and negative)
        None, -3, -2, -1, 0, 1, 2,
        # Single axis boundary cases
        -4, 3,
        # Multiple axes
        (-1, -2), (0, 1), (0, 2), (1, 2), (0, 1, 2),
        # Negative multiple axes
        (-3, -2), (-3, -1), (-2, -1)
    ])
    keepdims: List[bool] = field(default_factory=lambda: [
        True, False
    ])

var = InputSpace()