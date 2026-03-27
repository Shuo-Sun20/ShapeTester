import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union

def call_func(inputs, shape, dtype=None):
    return keras.ops.empty(shape, dtype)

valid_test_case = {
    "inputs": np.random.randn(3, 4, 5).astype('float32'),
    "shape": (3, 4, 5),
    "dtype": 'float32'
}

@dataclass
class InputSpace:
    shape: List[Union[Tuple[int, ...], int]] = field(default_factory=lambda: [
        (),
        (0,),
        (1,),
        (5,),
        (10, 0),
        (0, 10),
        (3, 4, 5),
        (2, 3, 4, 5),
        (1, 1, 1, 1, 1),
        (10, 10, 10),
        (2, 0, 5),
        (100, 1),
        (1, 100),
        (2, 3, 0, 4),
        (256, 256, 3),
        (1000,)
    ])