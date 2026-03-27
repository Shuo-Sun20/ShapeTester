import keras
from dataclasses import dataclass
from typing import List

valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4))
}

@dataclass
class InputSpace:
    pass