import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List

def call_func(inputs, ksize, strides, padding, data_format="NWC", name=None):
    return tf.nn.max_pool1d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

example_input = tf.constant(np.random.randn(2, 10, 3).astype(np.float32))
example_output = call_func(inputs=example_input, ksize=2, strides=2, padding="VALID")

valid_test_case = {
    "inputs": example_input,
    "ksize": 2,
    "strides": 2,
    "padding": "VALID",
    "data_format": "NWC",
    "name": None
}

@dataclass
class InputSpace:
    ksize: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    strides: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: List[Union[str, List[List[int]]]] = field(default_factory=lambda: [
        "VALID",
        "SAME",
        [[0, 0], [1, 1], [0, 0]],
        [[0, 0], [2, 2], [0, 0]],
        [[0, 0], [0, 0], [0, 0]]
    ])
    data_format: List[str] = field(default_factory=lambda: ["NWC", "NCW"])