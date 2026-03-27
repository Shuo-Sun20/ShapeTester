import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "base_operator": tf.linalg.LinearOperatorDiag(
        diag=[1.0, 2.0],
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True
    ),
    "u": tf.constant([[1.0], [2.0]]),
    "inputs": tf.constant([[1.0, 2.0], [3.0, 4.0]]),
    "diag_update": tf.constant([3.0]),
    "v": tf.constant([[1.0], [3.0]]),
    "is_diag_update_positive": None,
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "method": "matmul"
}

# Task 2 & 3: Define discretized value ranges for shape-affecting parameters
# (base_operator, u, diag_update, v)

# Example 1: No batch, M=2, N=2, K=1
base_op1 = tf.linalg.LinearOperatorDiag(diag=[1.0, 2.0])
u1 = tf.constant([[1.0], [2.0]])
diag_update1 = tf.constant([3.0])
v1 = tf.constant([[1.0], [3.0]])

# Example 2: Batch [2], M=2, N=2, K=1
base_op2 = tf.linalg.LinearOperatorDiag(diag=[[1.0, 2.0], [2.0, 3.0]])
u2 = tf.constant([[[1.0], [2.0]], [[2.0], [1.0]]])
diag_update2 = tf.constant([[3.0], [4.0]])
v2 = tf.constant([[[1.0], [3.0]], [[2.0], [1.0]]])

# Example 3: No batch, M=3, N=3, K=2, diag_update=None, v=None
base_op3 = tf.linalg.LinearOperatorDiag(diag=[1.0, 2.0, 3.0])
u3 = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
diag_update3 = None
v3 = None

# Example 4: Batch [2], M=3, N=3, K=2
base_op4 = tf.linalg.LinearOperatorDiag(diag=[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
u4 = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                  [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]])
diag_update4 = tf.constant([[3.0, 4.0], [5.0, 6.0]])
v4 = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                  [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]]])

# Example 5: Batch [2,2], M=2, N=2, K=1, v=None
base_op5 = tf.linalg.LinearOperatorDiag(diag=[[[1.0, 2.0], [2.0, 3.0]],
                                              [[3.0, 4.0], [4.0, 5.0]]])
u5 = tf.constant([[[[1.0], [2.0]], [[3.0], [4.0]]],
                  [[[2.0], [1.0]], [[4.0], [3.0]]]])
diag_update5 = tf.constant([[[3.0], [4.0]], [[5.0], [6.0]]])
v5 = None

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    base_operator: List[tf.linalg.LinearOperator] = field(
        default_factory=lambda: [base_op1, base_op2, base_op3, base_op4, base_op5]
    )
    u: List[tf.Tensor] = field(
        default_factory=lambda: [u1, u2, u3, u4, u5]
    )
    diag_update: List[Optional[tf.Tensor]] = field(
        default_factory=lambda: [diag_update1, diag_update2, diag_update3, diag_update4, diag_update5]
    )
    v: List[Optional[tf.Tensor]] = field(
        default_factory=lambda: [v1, v2, v3, v4, v5]
    )