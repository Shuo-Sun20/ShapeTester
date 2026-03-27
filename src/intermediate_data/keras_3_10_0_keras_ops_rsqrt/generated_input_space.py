import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

def call_func(inputs):
    return keras.ops.rsqrt(inputs)

# 1. Define valid_test_case
random_tensor = keras.ops.convert_to_tensor(np.random.uniform(0.1, 10.0, size=(5,)))
valid_test_case = {"inputs": random_tensor}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains parameters that affect output shape (except 'inputs').
    Since call_func() only has 'inputs' parameter and we're excluding it,
    the InputSpace class is empty.
    """
    pass