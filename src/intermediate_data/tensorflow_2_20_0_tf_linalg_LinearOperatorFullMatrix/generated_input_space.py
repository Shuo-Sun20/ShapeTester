import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=[2, 3, 5, 5]),
    "is_non_singular": True,
    "is_self_adjoint": False,
    "is_positive_definite": None,
    "is_square": None,
    "name": 'LinearOperatorFullMatrix'
}

# Task 4: Define InputSpace class with discretized value ranges
@dataclass
class InputSpace:
    # Only the inputs parameter affects the output shape
    inputs: List[tf.Tensor] = field(default_factory=lambda: [
        tf.random.normal(shape=[1, 1]),      # 1x1 matrix
        tf.random.normal(shape=[2, 2]),      # 2x2 matrix
        tf.random.normal(shape=[3, 5]),      # 3x5 matrix
        tf.random.normal(shape=[2, 3, 4]),   # 2 batches of 3x4 matrices
        tf.random.normal(shape=[2, 3, 5, 5]) # 2x3 batches of 5x5 matrices
    ])