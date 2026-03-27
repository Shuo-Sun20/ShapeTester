import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, axis=None, dtype=None):
    return keras.ops.cumprod(x=inputs, axis=axis, dtype=dtype)

np.random.seed(42)
random_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func(random_tensor, axis=0)

# Task 1: valid_test_case
valid_test_case = {
    "inputs": random_tensor,
    "axis": 0,
    "dtype": None
}

# Task 4: InputSpace dataclass
@dataclass
class InputSpace:
    axis: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, -1, -2, -3, 2])