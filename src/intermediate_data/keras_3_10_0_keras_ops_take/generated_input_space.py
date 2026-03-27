import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# Original function definition from the problem
def call_func(inputs, axis):
    x, indices = inputs
    return keras.ops.take(x, indices, axis)

# 1. Valid test case definition
x = keras.random.normal((3, 4, 5))
indices = keras.ops.convert_to_tensor([0, 2])
valid_test_case = {
    "inputs": [x, indices],
    "axis": 1
}

# 2 & 3. Parameters affecting output shape (besides inputs) and their value spaces
# Only "axis" parameter affects output shape besides the inputs

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # axis parameter value space
    # Includes: None (default), positive indices, negative indices
    # Boundary values: None, 0, ndim-1, -1, -ndim
    # Typical values for 3D tensor (3,4,5): None, 0, 1, 2, -1, -2, -3
    axis: List[Optional[int]] = field(
        default_factory=lambda: [None, 0, 1, 2, -1, -2, -3]
    )