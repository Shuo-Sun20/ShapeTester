import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [
        tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=tf.int32),
        tf.constant([0, 1, 0], dtype=tf.int32)
    ],
    'num_segments': 2,
    'name': None
}

# Tasks 2, 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    num_segments: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])