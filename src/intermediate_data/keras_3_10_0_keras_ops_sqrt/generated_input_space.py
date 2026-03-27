import keras
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "inputs": keras.random.uniform(shape=(2, 3), minval=0, maxval=10)
}

@dataclass
class InputSpace:
    pass