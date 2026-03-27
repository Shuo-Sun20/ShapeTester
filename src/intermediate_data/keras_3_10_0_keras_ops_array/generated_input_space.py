import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, dtype=None):
    return keras.ops.array(x=inputs, dtype=dtype)

valid_test_case = {"inputs": np.array([1.0, 2.0, 3.0]), "dtype": "float32"}

@dataclass
class InputSpace:
    # No parameters affect the shape of the output tensor except 'inputs'
    # Since 'inputs' is excluded, this dataclass remains empty
    pass