import keras
from dataclasses import dataclass, field
from typing import List, Union

def call_func(inputs, axis=0):
    return keras.ops.stack(x=inputs, axis=axis)

# Create example input tensors
tensor1 = keras.random.normal(shape=(2, 3))
tensor2 = keras.random.normal(shape=(2, 3))
tensor3 = keras.random.normal(shape=(2, 3))

# Valid test case
valid_test_case = {
    "inputs": [tensor1, tensor2, tensor3],
    "axis": 0
}

@dataclass
class InputSpace:
    axis: List[int] = field(default_factory=lambda: [
        -4,  # Boundary case for rank 2 tensors
        -3,  # Equivalent to axis=0 for rank 2 tensors
        -2,  # Equivalent to axis=1 for rank 2 tensors
        -1,  # Equivalent to axis=2 for rank 2 tensors
        0,   # Default value, valid for any tensor rank
        1,   # Valid for rank 2+ tensors
        2,   # Valid for rank 2+ tensors
        3,   # Boundary case for rank 2 tensors
    ])