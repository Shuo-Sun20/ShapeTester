import tensorflow as tf
from dataclasses import dataclass, field

# 1. valid_test_case definition
batch_shape = (10, 2, 2)
A = tf.random.normal(batch_shape, dtype=tf.float32)
A = tf.matmul(A, A, transpose_b=True) + tf.eye(2) * 1e-3
chol = tf.linalg.cholesky(A)
rhs = tf.random.normal((10, 2, 5), dtype=tf.float32)

valid_test_case = {
    'inputs': [chol, rhs],
    'name': None
}

# 2-4. InputSpace dataclass definition
@dataclass
class InputSpace:
    """Contains all parameters affecting output shape except 'inputs'"""
    name: list = field(default_factory=lambda: [None])