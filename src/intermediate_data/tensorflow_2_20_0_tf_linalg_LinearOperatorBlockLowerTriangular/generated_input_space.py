import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Any

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define valid_test_case variable
operator_00 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operator_10 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operator_11 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operators_2x2 = [[operator_00], [operator_10, operator_11]]

valid_test_case = {
    "operators": operators_2x2,
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "inputs": tf.random.normal(shape=[4, 3]),
    "adjoint": False,
    "adjoint_arg": False
}

# Create additional operator structures for value space
# 1x1 block structure
operator_1x1 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[3, 3]))
operators_1x1 = [[operator_1x1]]

# 3x3 block structure (2x2 blocks)
operator_00_3x3 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operator_10_3x3 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operator_11_3x3 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operator_20_3x3 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operator_21_3x3 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operator_22_3x3 = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2]))
operators_3x3 = [
    [operator_00_3x3],
    [operator_10_3x3, operator_11_3x3],
    [operator_20_3x3, operator_21_3x3, operator_22_3x3]
]

# 2x2 block structure with different block sizes
operator_00_mixed = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[3, 3]))
operator_10_mixed = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[4, 3]))
operator_11_mixed = tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[4, 4]))
operators_mixed = [[operator_00_mixed], [operator_10_mixed, operator_11_mixed]]

# Operators with different types
operator_diag = tf.linalg.LinearOperatorDiag(tf.random.normal(shape=[2]))
operator_lower = tf.linalg.LinearOperatorLowerTriangular(tf.random.normal(shape=[2, 2]))
operators_mixed_types = [[operator_diag], [operator_lower, operator_00]]

# Define InputSpace dataclass
@dataclass
class InputSpace:
    operators: List[Any] = field(default_factory=lambda: [
        operators_1x1,
        operators_2x2,
        operators_3x3,
        operators_mixed,
        operators_mixed_types
    ])
    adjoint: List[bool] = field(default_factory=lambda: [True, False])