import keras
import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple

# Define valid_test_case
valid_test_case = {
    "inputs": keras.random.uniform(
        shape=(4, 32, 32, 3),
        minval=0,
        maxval=255,
        dtype="float32"
    ),
    "factor": 0.2,
    "value_range": (0, 255),
    "data_format": None,
    "seed": None
}

# Parameters that can affect output shape: data_format
@dataclass
class InputSpace:
    data_format: list = None  # Can be None, "channels_last", or "channels_first"
    
    def __post_init__(self):
        if self.data_format is None:
            self.data_format = [None, "channels_last", "channels_first"]