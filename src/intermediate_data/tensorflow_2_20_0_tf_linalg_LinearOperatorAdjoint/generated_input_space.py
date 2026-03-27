import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
np.random.seed(42)
operator_matrix = np.random.randn(3, 3).astype(np.complex64)
input_matrix = np.random.randn(3, 4).astype(np.complex64)

valid_test_case = {
    'operator': tf.linalg.LinearOperatorFullMatrix(operator_matrix),
    'inputs': tf.constant(input_matrix),
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': None,
    'name': None
}

# 3. Value space for operator parameter
# Operator can be various LinearOperator types with different shapes
operator_instances = [
    # Square operators
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(2, 2).astype(np.float32)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(2, 2).astype(np.complex64)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(3, 3).astype(np.float32)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(3, 3).astype(np.complex64)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(5, 5).astype(np.float32)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(5, 5).astype(np.complex64)),
    
    # Rectangular operators
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(2, 3).astype(np.float32)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(2, 3).astype(np.complex64)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(3, 2).astype(np.float32)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(3, 2).astype(np.complex64)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(4, 6).astype(np.float32)),
    tf.linalg.LinearOperatorFullMatrix(np.random.randn(4, 6).astype(np.complex64)),
    
    # Diagonal operators
    tf.linalg.LinearOperatorDiag(np.random.randn(3).astype(np.float32)),
    tf.linalg.LinearOperatorDiag(np.random.randn(3).astype(np.complex64)),
    
    # Identity operators
    tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=tf.float32),
    tf.linalg.LinearOperatorIdentity(num_rows=3, dtype=tf.complex64),
    
    # Lower triangular operators
    tf.linalg.LinearOperatorLowerTriangular(np.random.randn(3, 3).astype(np.float32)),
    tf.linalg.LinearOperatorLowerTriangular(np.random.randn(3, 3).astype(np.complex64)),
    
    # The specific instance from valid_test_case
    valid_test_case['operator']
]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    operator: List[tf.linalg.LinearOperator] = field(
        default_factory=lambda: operator_instances
    )