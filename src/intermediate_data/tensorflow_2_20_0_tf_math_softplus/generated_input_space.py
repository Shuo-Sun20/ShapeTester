import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import Optional, List

def call_func(inputs, name=None):
    return tf.math.softplus(features=inputs, name=name)

# Task 1: valid_test_case
valid_test_case = {
    'inputs': tf.constant(np.random.randn(3, 4).astype(np.float32)),
    'name': None
}

# Task 2: Only 'inputs' affects output shape
# Task 3 & 4: InputSpace dataclass
@dataclass
class InputSpace:
    # Only 'inputs' affects output shape, so no other parameters to include
    pass

# The InputSpace class can be instantiated as: var = InputSpace()