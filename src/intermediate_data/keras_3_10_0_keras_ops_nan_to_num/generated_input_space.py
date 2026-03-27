import keras
import numpy as np
from dataclasses import dataclass, field

np.random.seed(42)
data = np.random.randn(3, 4).astype(np.float32)
data[0, 0] = np.nan
data[1, 1] = np.inf
data[2, 2] = -np.inf
x = keras.ops.convert_to_tensor(data)

valid_test_case = {
    'inputs': x,
    'nan': 0.0,
    'posinf': None,
    'neginf': None
}

@dataclass
class InputSpace:
    # Parameters that can affect the shape of the output tensor
    # (none for nan_to_num - all parameters only affect values, not shape)
    pass