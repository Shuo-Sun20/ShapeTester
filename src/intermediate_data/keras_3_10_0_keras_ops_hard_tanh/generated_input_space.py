import keras
import numpy as np
from dataclasses import dataclass

def call_func(inputs):
    return keras.ops.hard_tanh(inputs)

# 1. Valid test case
valid_test_case = {"inputs": np.random.uniform(-2, 2, size=(5,))}

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # No additional parameters affecting output shape beyond 'inputs'
    # Since keras.ops.hard_tanh() only takes 'x' (our 'inputs') as parameter
    pass