import keras
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4)),
    "fill_value": 5.0,
    "dtype": None
}

# Task 2: Identify parameters affecting output shape (excluding "inputs")
# Only "dtype" affects the output type but not shape. No parameters except "inputs" affect shape.

# Task 3 & 4: Define InputSpace with all shape-affecting parameters
@dataclass
class InputSpace:
    # Since no parameters except "inputs" affect shape, this dataclass is empty
    # But to make it instantiable and satisfy the requirement, we include a dummy field
    # Alternatively, we can simply define an empty dataclass that can be instantiated
    pass