import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, axis=-1):
    return keras.ops.argsort(x=inputs, axis=axis)

# 1. Valid test case
example_input = keras.ops.array(np.random.randint(0, 10, size=(3, 4)))
valid_test_case = {
    "inputs": example_input,
    "axis": 1
}

# 2. Parameters affecting output shape: axis

# 3. & 4. InputSpace dataclass with discretized value space for axis
@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [None, -2, -1, 0, 1])