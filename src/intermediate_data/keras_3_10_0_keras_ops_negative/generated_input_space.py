import keras
import numpy as np
from dataclasses import dataclass

def call_func(inputs):
    return keras.ops.negative(inputs)

x = keras.ops.convert_to_tensor(np.random.randn(2, 3))
example_output = call_func(x)

# Task 1: Define valid_test_case
valid_test_case = {"inputs": x}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # No parameters in call_func (other than "inputs") affect output shape
    pass