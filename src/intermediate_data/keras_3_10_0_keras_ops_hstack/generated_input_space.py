import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any

def call_func(inputs):
    return keras.ops.hstack(inputs)

# Generate tensors for valid_test_case
tensor1 = keras.random.normal(shape=(3, 4))
tensor2 = keras.random.normal(shape=(3, 2))
valid_test_case = {"inputs": [tensor1, tensor2]}

@dataclass
class InputSpace:
    """Parameters affecting output shape of keras.ops.hstack"""
    inputs: List[List[Any]] = field(default_factory=lambda: [
        # Boundary and typical test cases
        [keras.ops.zeros((0,))],  # empty 1D
        [keras.ops.ones((5,))],   # single 1D
        [keras.ops.ones((3,)), keras.ops.ones((3,))],  # multiple same shape 1D
        [keras.ops.ones((2,)), keras.ops.ones((3,))],  # different length 1D
        [keras.ops.ones((2,)), keras.ops.ones((2,)), keras.ops.ones((2,))],  # three 1D
        
        [keras.ops.zeros((3, 0))],  # empty columns 2D
        [keras.ops.ones((3, 4))],   # single 2D
        [keras.ops.ones((3, 2)), keras.ops.ones((3, 3))],  # same rows 2D
        [keras.ops.ones((0, 2)), keras.ops.ones((0, 3))],  # zero rows 2D
        [tensor1, tensor2],  # original test case
        
        # Higher dimensions
        [keras.ops.ones((2, 3, 4)), keras.ops.ones((2, 2, 4))],  # 3D tensors
        [keras.ops.ones((1, 2, 3, 4)), keras.ops.ones((1, 3, 3, 4))],  # 4D tensors
    ])