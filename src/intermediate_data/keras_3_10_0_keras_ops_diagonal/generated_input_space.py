import keras
from dataclasses import dataclass, field

def call_func(inputs, offset=0, axis1=0, axis2=1):
    return keras.ops.diagonal(inputs, offset, axis1, axis2)

example_input = keras.random.normal((2, 3, 4))
valid_test_case = {
    "inputs": example_input,
    "offset": 0,
    "axis1": 0,
    "axis2": 2
}

@dataclass
class InputSpace:
    offset: list = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    axis1: list = field(default_factory=lambda: [0, 1, 2])
    axis2: list = field(default_factory=lambda: [0, 1, 2])