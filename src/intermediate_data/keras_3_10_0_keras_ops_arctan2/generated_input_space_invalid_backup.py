import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List

def call_func(inputs):
    x1 = inputs[0]
    x2 = inputs[1]
    return keras.ops.arctan2(x1, x2)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        keras.ops.convert_to_tensor(np.random.randn(3, 4)),
        keras.ops.convert_to_tensor(np.random.randn(3, 4))
    ]
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    inputs: List = field(default_factory=lambda: [
        [
            keras.ops.convert_to_tensor(np.array([[1.0]])),
            keras.ops.convert_to_tensor(np.array([[1.0]]))
        ],
        [
            keras.ops.convert_to_tensor(np.random.randn(1, 3)),
            keras.ops.convert_to_tensor(np.random.randn(1, 3))
        ],
        [
            keras.ops.convert_to_tensor(np.random.randn(2, 2)),
            keras.ops.convert_to_tensor(np.random.randn(2, 2))
        ],
        [
            keras.ops.convert_to_tensor(np.random.randn(3, 1)),
            keras.ops.convert_to_tensor(np.random.randn(3, 1))
        ],
        [
            keras.ops.convert_to_tensor(np.random.randn(3, 4)),
            keras.ops.convert_to_tensor(np.random.randn(3, 4))
        ]
    ])