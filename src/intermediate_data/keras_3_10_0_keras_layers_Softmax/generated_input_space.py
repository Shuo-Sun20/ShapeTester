import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional
import keras

def call_func(axis=-1, inputs=None, mask=None):
    softmax_layer = keras.layers.Softmax(axis=axis)
    return softmax_layer(inputs, mask=mask)

valid_test_case = {
    "axis": -1,
    "inputs": np.random.randn(2, 5).astype(np.float32),
    "mask": None
}

@dataclass
class InputSpace:
    # Only the axis parameter can affect output shape (mask only masks values, doesn't change shape)
    axis: List[Union[int, List[int]]] = field(default_factory=lambda: [
        -1,  # Default case (last dimension)
        -2,  # Second-to-last dimension
        0,   # First dimension
        1,   # Second dimension
        [0, 1]  # Multiple dimensions for 2D input
    ])