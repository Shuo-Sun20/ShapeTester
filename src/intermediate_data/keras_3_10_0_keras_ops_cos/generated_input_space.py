import keras
import numpy as np
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4))
}

# Task 4: Define InputSpace class
@dataclass
class InputSpace:
    pass