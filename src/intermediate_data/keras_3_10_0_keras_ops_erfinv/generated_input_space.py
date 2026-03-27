import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs):
    return keras.ops.erfinv(inputs)

valid_test_case = {
    "inputs": np.random.uniform(-0.99, 0.99, (3, 4)).astype(np.float32)
}

@dataclass
class InputSpace:
    # The only parameter that affects output shape is 'inputs', 
    # but we are excluding 'inputs' per instructions
    pass