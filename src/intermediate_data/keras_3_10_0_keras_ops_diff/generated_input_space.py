import keras
import keras.ops as ops
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, n=1, axis=-1):
    a = inputs[0] if isinstance(inputs, list) else inputs
    return ops.diff(a, n=n, axis=axis)

# Valid test case
random_tensor = keras.random.uniform(shape=(5, 3))
valid_test_case = {
    "inputs": random_tensor,
    "n": 1,
    "axis": -1
}

# Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding "inputs")
    n: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])  # Boundary: min=1, max=5 (or tensor length)
    axis: List[int] = field(default_factory=lambda: [-2, -1, 0, 1])  # For 2D tensor (shape 5,3)

    # Optional: Include more axes for higher-dimensional tensors if needed
    # axis: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])

# Instantiate InputSpace
var = InputSpace()