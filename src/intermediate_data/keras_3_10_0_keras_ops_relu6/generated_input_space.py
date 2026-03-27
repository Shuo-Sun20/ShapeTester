import keras
import numpy as np
from dataclasses import dataclass
from typing import List

def call_func(inputs):
    return keras.ops.relu6(inputs)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(2, 3).astype(np.float32))
}

# Tasks 2, 3, and 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Since there are no parameters in call_func (except 'inputs' which is excluded)
    that affect the shape of the output, this class is empty.
    """
    pass