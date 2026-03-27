import tensorflow as tf
import numpy as np
from dataclasses import dataclass

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [tf.constant([[1.0, 2.0], [3.0, 0.0]], dtype=tf.float32)],
    "name": "test_operation"
}

# 2-4. Define InputSpace class
@dataclass
class InputSpace:
    # Only parameter in call_func that can affect output shape (besides inputs) is 'name',
    # but 'name' doesn't affect output shape. Therefore, no fields are needed for shape-affecting parameters.
    pass

# Note: The 'name' parameter doesn't affect output tensor shape - it only provides an optional operation name.
# The shape is solely determined by the input tensor 'x' from inputs[0], which is excluded by the question.