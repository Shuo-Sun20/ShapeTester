import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple

def call_func(inputs, size=(2, 2, 2), data_format=None):
    layer_instance = keras.layers.UpSampling3D(size=size, data_format=data_format)
    output = layer_instance(inputs)
    return output

valid_test_case = {
    "inputs": np.random.rand(2, 1, 2, 1, 3).astype('float32'),
    "size": (2, 2, 2),
    "data_format": "channels_last"
}

@dataclass
class InputSpace:
    size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            1,
            2,
            3,
            (1, 2, 3),
            (3, 3, 3)
        ]
    )
    data_format: List[str] = field(
        default_factory=lambda: [
            "channels_last",
            "channels_first"
        ]
    )