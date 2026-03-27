import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional

# 1. Valid test case
valid_test_case = {
    "inputs": [tf.random.normal(shape=[2, 2]), tf.random.normal(shape=[2, 2])],
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "name": None
}

# 2. & 3. Parameters affecting output shape (excluding inputs)
# Based on documentation and TensorFlow implementation, only 'inputs' affects output shape.
# The boolean hint parameters (is_non_singular, is_self_adjoint, is_positive_definite, is_square)
# and 'name' do NOT affect output shape.
# Therefore, InputSpace contains no parameters that affect shape.

@dataclass
class InputSpace:
    # Since no parameters except 'inputs' affect output shape, 
    # this class has no fields
    pass