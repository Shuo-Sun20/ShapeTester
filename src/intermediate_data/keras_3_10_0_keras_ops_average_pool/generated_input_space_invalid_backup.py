import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

def call_func(inputs, pool_size, strides=None, padding="valid", data_format=None):
    return keras.ops.average_pool(inputs, pool_size, strides, padding, data_format)

# Generate random input tensor for channels_last format (batch, height, width, channels)
batch_size, height, width, channels = 2, 8, 8, 3
input_tensor = keras.random.normal(shape=(batch_size, height, width, channels))
pool_size = (2, 2)
strides = (2, 2)

example_output = call_func(input_tensor, pool_size, strides, padding="valid", data_format="channels_last")

valid_test_case = {
    "inputs": input_tensor,
    "pool_size": (2, 2),
    "strides": (2, 2),
    "padding": "valid",
    "data_format": "channels_last"
}

@dataclass
class InputSpace:
    pool_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, 4, (2, 3)]
    )
    strides: List[Optional[Union[int, Tuple[int, int]]]] = field(
        default_factory=lambda: [None, 1, 2, 3, (1, 2)]
    )
    padding: List[str] = field(
        default_factory=lambda: ["valid", "same"]
    )
    data_format: List[str] = field(
        default_factory=lambda: ["channels_last", "channels_first"]
    )