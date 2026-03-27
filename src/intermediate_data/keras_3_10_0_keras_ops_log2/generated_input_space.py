import keras
import numpy as np

def call_func(inputs):
    return keras.ops.log2(inputs)

valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4))
}

from dataclasses import dataclass
from typing import Any, Union, List

@dataclass
class InputSpace:
    # Note: call_func only has one parameter "inputs", and we exclude it per instruction.
    # There are no additional parameters that affect output shape.
    pass