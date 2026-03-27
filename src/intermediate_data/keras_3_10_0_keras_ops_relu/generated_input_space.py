import keras
import numpy as np
from dataclasses import dataclass

# Task 1: Define valid_test_case
random_tensor = keras.random.normal(shape=(4, 3))
valid_test_case = {
    "inputs": [random_tensor]
}

# Task 2 & 3: Parameters affecting output shape (except "inputs")
# There are no additional parameters in call_func() beyond "inputs"
# that affect the output shape. The shape is solely determined by inputs[0].

# Task 4: Define InputSpace dataclass with all parameters affecting output shape
# Since there are no such parameters (except "inputs", which is excluded),
# the dataclass is empty but can still be instantiated.

@dataclass
class InputSpace:
    # No fields are defined because there are no parameters
    # beyond "inputs" that affect output shape
    pass