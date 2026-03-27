import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

def call_func(inputs, cropping=((0, 0), (0, 0)), data_format=None):
    layer = keras.layers.Cropping2D(cropping=cropping, data_format=data_format)
    return layer(inputs)

# 1. Valid test case
valid_test_case = {
    'inputs': np.random.randn(2, 28, 28, 3).astype(np.float32),
    'cropping': ((2, 2), (4, 4)),
    'data_format': None
}

@dataclass
class InputSpace:
    cropping: List[Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]] = field(
        default_factory=lambda: [0, 1, (1, 2), ((1, 2), (3, 4)), ((2, 2), (4, 4))]
    )
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )