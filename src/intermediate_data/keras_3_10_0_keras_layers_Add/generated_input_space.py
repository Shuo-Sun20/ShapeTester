import keras
import numpy as np
from dataclasses import dataclass

def call_func(inputs, name=None):
    add_layer = keras.layers.Add(name=name)
    return add_layer(inputs)

# Generate random tensors for valid_test_case
x1 = np.random.rand(2, 3, 4).astype('float32')
x2 = np.random.rand(2, 3, 4).astype('float32')

valid_test_case = {
    "inputs": [x1, x2],
    "name": "add_layer"
}

@dataclass
class InputSpace:
    pass