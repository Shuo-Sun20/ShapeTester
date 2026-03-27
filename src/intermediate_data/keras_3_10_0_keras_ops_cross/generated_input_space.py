import keras
import numpy as np
from dataclasses import dataclass, field

# Generate the same random input tensors as in the example
np.random.seed(42)
x1_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4, 3).astype(np.float32))
x2_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4, 3).astype(np.float32))

def call_func(inputs, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1, x2 = inputs
    return keras.ops.cross(x1, x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

valid_test_case = {
    'inputs': [x1_tensor, x2_tensor],
    'axis': -1
}

@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [None, -3, -2, -1, 0, 1, 2])