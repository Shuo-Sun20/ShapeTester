import keras.ops
import keras
from dataclasses import dataclass, field
from typing import Optional, Union, List

def call_func(inputs, axis=-1, order=2, epsilon=None):
    return keras.ops.normalize(x=inputs, axis=axis, order=order, epsilon=epsilon)

x = keras.random.uniform(shape=(2, 3, 4))
valid_test_case = {
    "inputs": x,
    "axis": -1,
    "order": 2,
    "epsilon": None
}

@dataclass
class InputSpace:
    axis: List[Union[int, tuple]] = field(default_factory=lambda: [-1, 0, 1, (0, 1), (1, 2), None])