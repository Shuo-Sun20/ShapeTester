import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Define valid_test_case
batch_size = 2
M = 3
K = 4
matrix_tensor = tf.random.normal(shape=[batch_size, M, M], dtype=tf.float32)
rhs_tensor = tf.random.normal(shape=[batch_size, M, K], dtype=tf.float32)

valid_test_case = {
    'inputs': [matrix_tensor, rhs_tensor],
    'adjoint': False,
    'name': None
}

# Task 4: Define InputSpace
@dataclass
class InputSpace:
    adjoint: List[bool] = field(default_factory=lambda: [True, False])
    name: List[Optional[str]] = field(default_factory=lambda: [None, "solve1", "solve2", "solve3", "solve4"])