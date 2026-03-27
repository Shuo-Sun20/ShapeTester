import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(inputs, l2_regularizer=0.0, fast=True, name=None):
    matrix, rhs = inputs[0], inputs[1]
    output = tf.linalg.lstsq(matrix, rhs, l2_regularizer=l2_regularizer, fast=fast, name=name)
    return output

# 1. Define valid_test_case dictionary
matrix = tf.random.normal(shape=[2, 5, 3])
rhs = tf.random.normal(shape=[2, 5, 2])
valid_test_case = {
    "inputs": [matrix, rhs],
    "l2_regularizer": 0.0,
    "fast": True,
    "name": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape: l2_regularizer and fast
    # l2_regularizer is continuous; boundary values and 5 typical values
    l2_regularizer: List[float] = field(default_factory=lambda: [
        0.0,  # No regularization
        1e-8, # Very small regularization
        1e-4, # Small regularization
        0.1,  # Medium regularization
        1.0   # Strong regularization
    ])
    # fast is discrete with only 2 possible values
    fast: List[bool] = field(default_factory=lambda: [True, False])