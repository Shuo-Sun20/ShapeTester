import numpy as np
import keras.ops as ops
from dataclasses import dataclass
from typing import List

# Generate random 3x3 matrix for valid test case
np.random.seed(42)
x = ops.convert_to_tensor(np.random.randn(3, 3).astype(np.float32))

valid_test_case = {
    "inputs": [x]
}

@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output shape (except 'inputs')"""
    # Note: The only parameter in call_func is "inputs", which contains the matrix.
    # However, the shape of the output of slogdet is always two scalars regardless of input matrix shape,
    # provided the input is 2D and square. Since we are excluding "inputs" parameter itself,
    # there are no other parameters in call_func that affect output shape.
    pass