import keras
import numpy as np
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(
        np.random.uniform(-10.0, 10.0, size=(3, 5))
    )
}

# Task 2, 3, 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # There are no parameters other than "inputs" that affect output shape
    pass