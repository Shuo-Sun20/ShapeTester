import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# Existing code from the snippet
def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.var(inputs, axis=axis, keepdims=keepdims)

random_tensor = keras.random.normal(shape=(3, 4, 5))

# 1. Define valid_test_case
valid_test_case = {
    "inputs": random_tensor,
    "axis": 1,
    "keepdims": True
}

# 2 & 3. Parameters affecting output shape: axis and keepdims
# Value space for keepdims (boolean)
keepdims_values = [True, False]

# Value space for axis
# For a 3D tensor of shape (3, 4, 5):
# - None: compute variance of flattened tensor
# - int: single axis (positive: 0, 1, 2; negative: -1, -2, -3)
# - tuple: multiple axes combinations
# - empty tuple: no reduction (returns same shape as input when keepdims=True)
axis_values = [
    None,           # default - flattened
    0, 1, 2,        # positive axes
    -1, -2, -3,     # negative axes
    (0, 1), (0, 2), (1, 2), (0, 1, 2),  # multiple axes
    (-1, -2), (-1, -3), (-2, -3), (-1, -2, -3),  # negative multiple
    ()              # empty tuple - no reduction
]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    axis: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,
            0, 1, 2,
            -1, -2, -3,
            (0, 1), (0, 2), (1, 2), (0, 1, 2),
            (-1, -2), (-1, -3), (-2, -3), (-1, -2, -3),
            ()
        ]
    )
    keepdims: List[bool] = field(default_factory=lambda: [True, False])

# Example instantiation
var = InputSpace()