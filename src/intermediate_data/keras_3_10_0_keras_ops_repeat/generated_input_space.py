import keras
import numpy as np
from dataclasses import dataclass, field

# 1. Define valid_test_case
valid_test_case = {
    'inputs': keras.random.normal(shape=(3, 4)),
    'repeats': 2,
    'axis': 1
}

def call_func(inputs, repeats, axis=None):
    return keras.ops.repeat(x=inputs, repeats=repeats, axis=axis)

# 2-4. Define InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    # repeats: integer or 1D array/list of integers
    # Discretized value space covering:
    # - Boundary values (0, 1, 10)
    # - Typical small values (2, 3)
    # - Medium values (5)
    # - Array inputs (list format)
    repeats: list = field(default_factory=lambda: [
        0,  # boundary: no repetition
        1,  # boundary: single repetition (no change in count)
        2,  # valid_test_case value
        3,  # typical small value
        5,  # typical medium value
        10, # larger value
        [0, 1, 2, 3],  # array input with varying repeats
        [1, 1, 1, 1],  # uniform array input
        [2, 2, 2, 2],  # uniform array input (matching valid_test_case)
        [0, 0, 0, 0],  # all zeros (removes elements)
        [1, 2, 3, 4],  # ascending pattern
        [4, 3, 2, 1]   # descending pattern
    ])
    
    # axis: integer or None
    # Discretized value space covering:
    # - None (default, flatten array)
    # - Valid axis indices for 2D tensor (0, 1, -1, -2)
    # - Boundary invalid values to test error handling
    axis: list = field(default_factory=lambda: [
        None,  # default (flattened)
        0,     # first axis
        1,     # second axis (valid_test_case value)
        -1,    # last axis (same as 1 for 2D)
        -2,    # second last axis (same as 0 for 2D)
        2,     # out-of-bounds for 2D (for error testing)
        -3     # out-of-bounds for 2D (for error testing)
    ])

# Example instantiation
var = InputSpace()