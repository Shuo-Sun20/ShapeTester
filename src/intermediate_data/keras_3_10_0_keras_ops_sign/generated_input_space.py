import keras
from dataclasses import dataclass
from typing import List, Union
import numpy as np

def call_func(inputs):
    return keras.ops.sign(inputs)

# 1. valid_test_case definition
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor([[-1.5, 0.0, 2.3], [0.0, -0.0, 4.1]])
}

# 2. Parameters affecting output shape (excluding "inputs"): None
# The shape of the output tensor is solely determined by the shape of the input tensor.
# No other parameters exist in call_func().

# 3. Since no other parameters exist, no value spaces need to be constructed.

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # Since there are no parameters other than "inputs" that affect the output shape,
    # the class contains no fields.
    pass

# Example instantiation
var = InputSpace()