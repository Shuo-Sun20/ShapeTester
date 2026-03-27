import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

def call_func(inputs):
    return keras.ops.copy(inputs)

example_input = keras.random.normal(shape=(3, 4))
valid_test_case = {"inputs": example_input}

@dataclass
class InputSpace:
    # Since call_func only has one parameter "inputs" and we're excluding it,
    # there are no additional parameters affecting output shape
    # This is an empty dataclass by design
    pass