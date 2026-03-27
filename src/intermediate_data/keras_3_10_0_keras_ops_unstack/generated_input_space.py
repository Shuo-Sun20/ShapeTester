import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional

def call_func(inputs, axis=0, num=None):
    return keras.ops.unstack(inputs, axis=axis, num=num)

x = keras.ops.array(np.random.randn(3, 4, 5))
example_output = call_func(x, axis=1)

valid_test_case = {
    "inputs": x,
    "axis": 1,
    "num": None
}

@dataclass
class InputSpace:
    """
    Data class containing parameters that affect the output shape of keras.ops.unstack.
    Excludes 'inputs' parameter.
    """
    axis: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])
    num: List[Optional[int]] = field(default_factory=lambda: [None, 1, 2, 3, 4, 5])

var = InputSpace()