import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

# Task 1: Define valid_test_case variable
valid_test_case = {
    'inputs': keras.random.uniform(shape=(3, 4)) > 0.5,
    'axis': 1,
    'keepdims': True
}

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.any(x=inputs, axis=axis, keepdims=keepdims)

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    axis: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,
            0,
            (0, 1),
            -1,
            (0, -1)
        ]
    )
    keepdims: List[bool] = field(
        default_factory=lambda: [False, True]
    )