import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.sum(x=inputs, axis=axis, keepdims=keepdims)

# Construct valid input for test case
random_tensor = keras.ops.convert_to_tensor(np.random.rand(3, 4, 5))
valid_test_case = {
    'inputs': random_tensor,
    'axis': 1,
    'keepdims': True
}

@dataclass
class InputSpace:
    axis: List[Union[int, Tuple[int, ...], List[int], None]] = field(
        default_factory=lambda: [
            None,
            # Single axis values
            0, 1, 2, -1, -2, -3,
            # Boundary cases for 3D tensor
            0, 2,  # First and last positive indices
            -3, -1,  # First and last negative indices
            # Tuple axes
            (0, 1), (0, 2), (1, 2), (0, 1, 2),
            (-1, -2), (-1, -3), (-2, -3), (-3, -2, -1),
            # Mixed positive/negative
            (0, -1), (1, -2),
            # Empty tuple (should behave like None)
            (),
            # List axes
            [0, 1], [0, -1],
        ]
    )
    keepdims: List[bool] = field(
        default_factory=lambda: [True, False]
    )