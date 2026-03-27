import keras
from dataclasses import dataclass, field

def call_func(inputs, axes=2):
    x1, x2 = inputs
    return keras.ops.tensordot(x1, x2, axes=axes)

x1 = keras.random.normal(shape=(3, 4, 5))
x2 = keras.random.normal(shape=(5, 4, 3))
valid_test_case = {
    "inputs": [x1, x2],
    "axes": 1
}

@dataclass
class InputSpace:
    axes: list = field(default_factory=lambda: [
        0,
        1,
        2,
        3,
        [[0, 1], [1, 2]],
        [[0, 2], [2, 0]],
        [[1, 2], [0, 1]],
        [[0, 1, 2], [0, 1, 2]],
        [[], []]
    ])