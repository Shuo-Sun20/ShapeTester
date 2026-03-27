import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs):
    x, y = inputs
    return keras.ops.bitwise_left_shift(x, y)

x_tensor = keras.ops.convert_to_tensor(np.random.randint(0, 100, size=(3, 3)))
y_tensor = keras.ops.convert_to_tensor(np.random.randint(0, 8, size=(3, 3)))
valid_test_case = {'inputs': [x_tensor, y_tensor]}

@dataclass
class InputSpace:
    # Only parameter of call_func except 'inputs' is none
    pass