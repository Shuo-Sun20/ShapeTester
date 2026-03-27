import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List

def call_func(inputs, alpha=1.0, name=None, dtype=None):
    elu_instance = keras.layers.ELU(alpha=alpha, name=name, dtype=dtype)
    return elu_instance(inputs)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.randn(2, 5, 10).astype(np.float32),
    "alpha": 1.0,
    "name": None,
    "dtype": None
}

# Tasks 2, 3, 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Parameters that affect output shape in call_func (excluding 'inputs').
    Based on analysis: None of the parameters (alpha, name, dtype) affect output shape.
    The ELU layer always returns output with same shape as input.
    """
    # Note: No fields are defined since no parameters affect output shape
    # This maintains the requirement that var=InputSpace() can be instantiated
    pass