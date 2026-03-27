import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    "operator": tf.linalg.LinearOperatorFullMatrix(tf.random.normal(shape=[2, 2])),
    "inputs": tf.random.normal(shape=[2, 4]),
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "name": None
}

@dataclass
class InputSpace:
    operator: List[tf.linalg.LinearOperator] = field(default_factory=lambda: [
        tf.linalg.LinearOperatorFullMatrix(tf.eye(1)),
        tf.linalg.LinearOperatorFullMatrix(tf.eye(2)),
        tf.linalg.LinearOperatorFullMatrix(tf.eye(3)),
        tf.linalg.LinearOperatorFullMatrix(tf.eye(4)),
        tf.linalg.LinearOperatorFullMatrix(tf.eye(5))
    ])