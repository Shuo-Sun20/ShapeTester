import tensorflow as tf
import keras
from dataclasses import dataclass, field
from typing import Union, List, Optional

# Provided call_func
def call_func(inputs, scale, offset=0.0, name=None, dtype=None):
    layer = keras.layers.Rescaling(scale=scale, offset=offset, name=name, dtype=dtype)
    return layer(inputs)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": tf.random.uniform(shape=(2, 128, 128, 3), minval=0, maxval=255, dtype=tf.float32),
    "scale": 1.0 / 255.0,
    "offset": 0.0,
    "name": "test_rescaling",
    "dtype": None
}

# Task 2 & 3: Parameters affecting output shape - NONE (Rescaling preserves shape)
# Task 4: Define InputSpace dataclass with empty parameters (no shape-affecting parameters)
@dataclass
class InputSpace:
    # No parameters affect output shape for Rescaling layer
    pass

# Test instantiation
var = InputSpace()