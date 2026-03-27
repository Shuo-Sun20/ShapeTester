import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, pool_size, strides=None, padding="valid", data_format=None):
    output = keras.ops.max_pool(inputs, pool_size, strides, padding, data_format)
    return output

input_tensor = keras.random.normal((2, 5, 5, 3))
valid_test_case = {
    "inputs": input_tensor,
    "pool_size": (2, 2),
    "strides": (2, 2),
    "padding": "valid",
    "data_format": "channels_last"
}

@dataclass
class InputSpace:
    pool_size: list = field(default_factory=lambda: [1, 2, 3, 4, (2, 2), (2, 3), (3, 2)])
    strides: list = field(default_factory=lambda: [1, 2, 3, 4, (1, 2), (2, 1), (2, 2)])
    padding: list = field(default_factory=lambda: ["valid", "same"])
    data_format: list = field(default_factory=lambda: ["channels_last", "channels_first"])