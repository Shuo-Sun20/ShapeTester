import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, List

# Valid test case for call_func
valid_test_case = {
    "inputs": tf.constant([[0.1, 0.2, 0.3], [-0.4, -0.5, 0.6]], dtype=tf.float32),
    "name": "erfinv_operation"
}

# Definition of InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape (excluding 'inputs'):
    # Only 'name' exists, but it doesn't affect output shape.
    # Since there are no parameters that affect shape (other than inputs),
    # we define an empty dataclass with a placeholder field to meet instantiation requirements.
    pass