import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Union, List, Tuple

def call_func(axis, inputs):
    if isinstance(inputs, list):
        input_tensor = inputs[0] if len(inputs) == 1 else inputs
    else:
        input_tensor = inputs
    
    layer = keras.layers.UnitNormalization(axis=axis)
    output_tensor = layer(input_tensor)
    return output_tensor

valid_test_case = {
    "axis": -1,
    "inputs": np.random.randn(3, 4, 5).astype(np.float32)
}

@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [-3, -1, 0, 2, (-1, -2)])