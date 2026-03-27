import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(f, inputs, reverse=False, axis=0):
    return keras.ops.associative_scan(f, inputs, reverse=reverse, axis=axis)

# 1. Valid test case
xs = keras.ops.convert_to_tensor(np.random.rand(5))
sum_fn = lambda x, y: x + y
valid_test_case = {
    "f": sum_fn,
    "inputs": xs,
    "reverse": False,
    "axis": 0
}

# 4. InputSpace class with discretized value ranges
@dataclass
class InputSpace:
    reverse: list[bool] = field(default_factory=lambda: [False, True])
    axis: list[int] = field(default_factory=lambda: [-4, -1, 0, 1, 3])