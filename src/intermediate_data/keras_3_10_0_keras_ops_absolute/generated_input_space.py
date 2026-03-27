import keras
import numpy as np
from dataclasses import dataclass

valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4))
}

@dataclass
class InputSpace:
    pass