import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Define valid_test_case
op1_matrix = tf.random.normal(shape=[2, 2], seed=0)
op2_matrix = tf.random.normal(shape=[2, 2], seed=1)
operator_1 = tf.linalg.LinearOperatorFullMatrix(op1_matrix)
operator_2 = tf.linalg.LinearOperatorFullMatrix(op2_matrix)
operators = [operator_1, operator_2]
inputs = tf.random.normal(shape=[4, 2], seed=2)

valid_test_case = {
    "operators": operators,
    "inputs": inputs,
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "name": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    operators: List[List[tf.linalg.LinearOperator]] = field(
        default_factory=lambda: [
            [tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2], seed=3))],
            [tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[3, 3], seed=4))],
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2], seed=5)),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2], seed=6))
            ],
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 3], seed=7)),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[3, 2], seed=8))
            ],
            [
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2], seed=9)),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2], seed=10)),
                tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2], seed=11))
            ]
        ]
    )