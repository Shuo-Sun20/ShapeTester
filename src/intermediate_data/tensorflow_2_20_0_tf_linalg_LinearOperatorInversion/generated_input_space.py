import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

# Fixed random seed for reproducibility
tf.random.set_seed(42)

# Define the operators that will be used in valid_test_case and InputSpace
matrix_2x2 = tf.random.normal(shape=[2, 2])
operator_2x2 = tf.linalg.LinearOperatorFullMatrix(matrix_2x2)

matrix_3x3 = tf.random.normal(shape=[3, 3])
operator_3x3 = tf.linalg.LinearOperatorFullMatrix(matrix_3x3)

valid_test_case = {
    "operator": operator_2x2,
    "inputs": tf.random.normal(shape=[2, 4]),
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "name": None
}

@dataclass
class InputSpace:
    operator: List[tf.linalg.LinearOperator] = field(
        default_factory=lambda: [
            operator_2x2,
            operator_3x3,
            tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[1, 1])),
            tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[4, 4])),
            tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[5, 5])),
            tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[6, 6]))
        ]
    )