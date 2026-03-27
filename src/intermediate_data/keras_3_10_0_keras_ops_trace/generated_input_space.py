import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, offset=0, axis1=0, axis2=1):
    return keras.ops.trace(x=inputs, offset=offset, axis1=axis1, axis2=axis2)

# Construct a random 4x4 tensor as input
random_tensor = keras.random.normal(shape=(4, 4))
valid_test_case = {
    "inputs": random_tensor,
    "offset": 0,
    "axis1": 0,
    "axis2": 1
}

@dataclass
class InputSpace:
    axis1: List[int] = field(default_factory=lambda: [-4, -2, 0, 1, 2, 3])
    axis2: List[int] = field(default_factory=lambda: [-4, -2, 0, 1, 2, 3])