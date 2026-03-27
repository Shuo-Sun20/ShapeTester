import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32),
    'full_matrices': False,
    'compute_uv': True,
    'name': None
}

# Tasks 2-4: Define InputSpace class with shape-affecting parameters
@dataclass
class InputSpace:
    full_matrices: List[bool] = field(default_factory=lambda: [True, False])
    compute_uv: List[bool] = field(default_factory=lambda: [True, False])