from dataclasses import dataclass
import keras
import numpy as np

valid_test_case = {
    "inputs": np.random.randn(3, 4, 5).astype(np.float32),
    "axis": 1,
    "keepdims": True
}

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.median(x=inputs, axis=axis, keepdims=keepdims)

@dataclass
class InputSpace:
    axis: list = None
    keepdims: list = None

    def __post_init__(self):
        if self.axis is None:
            # Discrete parameter space for axis (including boundary and typical values)
            self.axis = [None, 0, -1, (0, 1), (1, -1), -2, 1, (0, 2)]
        if self.keepdims is None:
            # Discrete parameter space for keepdims (boolean)
            self.keepdims = [True, False]