import keras
import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple, List

valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(3, 4, 5)),
    "axis": 1
}

def call_func(inputs, axis=None):
    return keras.ops.flip(x=inputs, axis=axis)

@dataclass
class InputSpace:
    axis: List[Union[None, int, Tuple[int, ...]]] = None
    
    def __post_init__(self):
        if self.axis is None:
            self.axis = [
                None,
                0,
                1,
                2,
                (0, 1),
                (0, 2),
                (1, 2),
                (0, 1, 2),
                -1,
                -2,
                -3,
                (-1, -2),
                (-1, -3),
                (-2, -3),
                (0, -1),
                (1, -2),
                100,  # For negative testing (out-of-bound)
                "invalid",  # For negative testing (wrong type)
                [0, 1],  # For negative testing (list vs tuple)
                (0.5,),  # For negative testing (float axis)
            ]

# Instantiation example
var = InputSpace()