import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, num_classes, axis=-1, dtype=None, sparse=False):
    return keras.ops.multi_hot(
        inputs=inputs,
        num_classes=num_classes,
        axis=axis,
        dtype=dtype,
        sparse=sparse
    )

data = keras.ops.convert_to_tensor(np.random.randint(0, 5, size=10))
valid_test_case = {
    "inputs": data,
    "num_classes": 5,
    "axis": -1,
    "dtype": None,
    "sparse": False
}

@dataclass
class InputSpace:
    # Parameters that affect output tensor shape
    num_classes: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])
    axis: List[int] = field(default_factory=lambda: [-4, -3, -2, -1, 0, 1, 2, 3])