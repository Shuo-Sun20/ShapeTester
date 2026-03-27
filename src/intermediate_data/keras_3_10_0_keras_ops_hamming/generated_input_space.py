import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs):
    return keras.ops.hamming(inputs)

# Task 1: Define valid_test_case
valid_test_case = {"inputs": keras.ops.convert_to_tensor(5)}

# Task 4: Define InputSpace class with parameter value ranges
@dataclass
class InputSpace:
    inputs: list = field(default_factory=lambda: [1, 2, 3, 5, 7, 10, 20, 50, 100])