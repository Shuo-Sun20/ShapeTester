import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional

# Task 1: Define valid_test_case
N = 3
col = tf.constant([1.0, 2.0, 3.0])
row = tf.constant([1.0, 4.0, -9.0])
x = tf.random.normal(shape=[N, 2])

valid_test_case = {
    'inputs': [col, row, x],
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': None,
    'name': 'toeplitz_operator'
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape (excluding inputs)
    is_non_singular: Optional[bool] = field(default_factory=lambda: [None, True, False])
    is_self_adjoint: Optional[bool] = field(default_factory=lambda: [None, True, False])
    is_positive_definite: Optional[bool] = field(default_factory=lambda: [None, True, False])
    is_square: Optional[bool] = field(default_factory=lambda: [None, True, False])
    name: str = field(default_factory=lambda: [None, 'toeplitz_operator', 'test_op', ''])