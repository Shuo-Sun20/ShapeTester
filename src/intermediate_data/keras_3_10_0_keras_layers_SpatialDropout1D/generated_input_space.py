import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional

# Task 1: Define valid_test_case
input_tensor = keras.random.normal(shape=(2, 10, 5))
valid_test_case = {
    "rate": 0.5,
    "inputs": input_tensor,
    "seed": 42,
    "name": "test_dropout",
    "dtype": None,
    "training": True
}

# Task 2-4: Define InputSpace dataclass
# Parameters that can affect output shape (besides inputs): None
# SpatialDropout1D does not change output shape regardless of parameters
@dataclass
class InputSpace:
    pass