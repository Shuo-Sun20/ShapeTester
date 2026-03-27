import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, axis=0):
    return keras.ops.concatenate(xs=inputs, axis=axis)

tensor1 = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))
tensor2 = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))
tensor3 = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))

valid_test_case = {'inputs': [tensor1, tensor2, tensor3], 'axis': 0}

@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [-3, -1, 0, 1, 2])