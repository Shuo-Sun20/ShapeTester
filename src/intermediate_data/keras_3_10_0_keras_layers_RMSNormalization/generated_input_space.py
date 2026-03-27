import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

def call_func(axis, epsilon, inputs):
    layer = keras.layers.RMSNormalization(axis=axis, epsilon=epsilon)
    return layer(inputs)

random_tensor = np.random.rand(1, 10).astype(np.float32)
valid_test_case = {
    "axis": -1,
    "epsilon": 1e-06,
    "inputs": random_tensor
}

@dataclass
class InputSpace:
    axis: List[Union[int, tuple]] = field(default_factory=lambda: [
        -1,
        0,
        1,
        2,
        (0, 1)
    ])