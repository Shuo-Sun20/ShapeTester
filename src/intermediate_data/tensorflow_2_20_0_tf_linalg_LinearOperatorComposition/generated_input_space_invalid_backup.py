import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(operators, is_non_singular=None, is_self_adjoint=None,
              is_positive_definite=None, is_square=None, name=None, inputs=None):
    linear_operator = tf.linalg.LinearOperatorComposition(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name
    )
    return linear_operator.matmul(inputs)

# Create a valid test case
matrix1 = tf.random.normal(shape=[2, 3])
matrix2 = tf.random.normal(shape=[3, 4])
operator1 = tf.linalg.LinearOperatorFullMatrix(matrix1)
operator2 = tf.linalg.LinearOperatorFullMatrix(matrix2)
input_tensor = tf.random.normal(shape=[4, 5])

valid_test_case = {
    "operators": [operator1, operator2],
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "name": None,
    "inputs": input_tensor
}

@dataclass
class InputSpace:
    # Only 'operators' affects output shape (excluding 'inputs')
    operators: List[List[tf.linalg.LinearOperator]] = field(
        default_factory=lambda: [
            # Case 1: Basic 2D composition
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 3])),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[3, 4]))
            ],
            # Case 2: Single operator
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[5, 6]))
            ],
            # Case 3: Three operators
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 3])),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[3, 4])),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[4, 5]))
            ],
            # Case 4: Batch operators (2x3 batch of 4x5 and 5x6)
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 3, 4, 5])),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 3, 5, 6]))
            ],
            # Case 5: Square operators (2x2 composition)
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2])),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
            ]
        ]
    )