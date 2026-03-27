import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, name=None):
    x = inputs[0]
    y = inputs[1]
    return tf.math.xlog1py(x, y, name)

# 1. Valid test case
x = tf.constant(np.random.randn(2, 3), dtype=tf.float32)
y = tf.constant(np.random.randn(2, 3), dtype=tf.float32)
valid_test_case = {
    "inputs": [x, y],
    "name": "test_operation"
}

# 2. Parameters affecting output shape (excluding "inputs")
# Only parameter is "name" which doesn't affect shape.
# The shape is determined by broadcasting rules between x and y tensors,
# but these are passed through "inputs" parameter.

# 3. Value space analysis for parameters (excluding "inputs"):
# - name: string type, doesn't affect shape, typical values include None or operation names

# 4. InputSpace dataclass with all parameters affecting shape
# Since only "name" parameter remains and it doesn't affect shape,
# InputSpace would contain only the "name" parameter with its value space
# However, based on tensorflow's xlog1py behavior, no parameters besides "inputs" affect shape

@dataclass
class InputSpace:
    """Contains all parameters of call_func that affect output tensor shape."""
    
    # Since only "name" parameter exists and it doesn't affect shape,
    # we'll still include it for completeness, but note shape isn't affected
    name: List[str] = field(default_factory=lambda: [None, "test_operation", "xlog1py_op", "custom_name", ""])
    
    # Note: The "inputs" parameter contains x and y tensors which determine shape,
    # but we're excluding it per instructions

# Example instantiation
var = InputSpace()