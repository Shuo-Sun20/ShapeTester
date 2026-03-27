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
    # size parameter: can be int or tuple of 3 ints, must be >=1
    size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            1,  # minimal upsampling (no change)
            2,  # default integer size
            3,  # typical size
            5,  # larger size
            10, # boundary/large size
            (1, 1, 1),  # minimal tuple
            (2, 2, 2),  # default tuple
            (3, 3, 3),  # symmetric tuple
            (2, 3, 4),  # asymmetric tuple
            (1, 5, 10), # mixed boundary tuple
            (10, 1, 1)  # another mixed tuple
        ]
    )
    # data_format parameter: can be None, "channels_last", or "channels_first"
    data_format: List[Union[None, str]] = field(
        default_factory=lambda: [
            None,  # defaults to keras config
            "channels_last",  # default format
            "channels_first"  # alternative format
        ]
    )