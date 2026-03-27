import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, data_format=None, keepdims=False):
    layer = keras.layers.GlobalAveragePooling2D(data_format=data_format, keepdims=keepdims)
    return layer(inputs)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.rand(2, 4, 5, 3).astype(np.float32),
    "data_format": None,
    "keepdims": False
}

# Task 2 & 3: Parameters affecting output shape (except "inputs") and their value spaces
# - data_format: Discrete, possible values [None, "channels_last", "channels_first"]
# - keepdims: Discrete, possible values [True, False]

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    keepdims: List[bool] = field(default_factory=lambda: [True, False])

# Example instantiation (not required in output but demonstrating it works)
var = InputSpace()