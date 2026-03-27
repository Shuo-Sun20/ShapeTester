import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.randn(2, 32, 32, 3).astype(np.float32),
    "training": True,
    "mode": "horizontal_and_vertical",
    "seed": 42,
    "data_format": "channels_last",
    "name": "random_flip_layer",
    "dtype": "float32"
}

# Task 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that can affect the shape of
    the output tensor from call_func, with discretized value ranges.
    """
    # None of the parameters except 'inputs' affect output shape
    pass