import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, axis1, axis2):
    return keras.ops.swapaxes(inputs, axis1, axis2)

tensor = keras.random.normal(shape=(3, 4, 5))
valid_test_case = {
    "inputs": tensor,
    "axis1": 0,
    "axis2": 2
}

@dataclass
class InputSpace:
    axis1: list = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])
    axis2: list = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])