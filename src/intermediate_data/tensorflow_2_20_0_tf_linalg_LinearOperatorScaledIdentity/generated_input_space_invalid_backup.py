import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'num_rows': 5,
    'multiplier': tf.constant([2.0, 3.0], dtype=tf.float32),
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': True,
    'name': 'ScaledIdentityOperator',
    'inputs': [tf.random.normal(shape=[2, 5, 4], dtype=tf.float32)]
}

@dataclass
class InputSpace:
    num_rows: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    multiplier: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant(0.0, dtype=tf.float32),
        tf.constant(1.0, dtype=tf.float32),
        tf.constant(-1.0, dtype=tf.float32),
        tf.constant([0.0, 1.0], dtype=tf.float32),
        tf.constant([-2.0, 2.0], dtype=tf.float32)
    ])