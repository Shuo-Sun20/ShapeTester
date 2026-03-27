import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List

# 1. Define valid_test_case
valid_test_case = {
    'operator': tf.linalg.LinearOperatorFullMatrix(
        np.random.randn(3, 3).astype(np.complex64)
    ),
    'inputs': tf.constant(np.random.randn(3, 4).astype(np.complex64)),
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': None,
    'name': None
}

# 2. Parameters affecting output shape: operator

# 3. Value space for operator
np.random.seed(42)
operator_values = [
    # Small square matrix
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(2, 2).astype(np.complex64)),
    # Medium square matrix
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(3, 3).astype(np.complex64)),
    # Large square matrix
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(4, 4).astype(np.complex64)),
    # Tall matrix (more rows)
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(5, 3).astype(np.complex64)),
    # Wide matrix (more columns)
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(3, 5).astype(np.complex64))
]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    operator: List[tf.linalg.LinearOperator] = field(default_factory=lambda: operator_values)