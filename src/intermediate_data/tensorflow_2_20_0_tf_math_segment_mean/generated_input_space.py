import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": tf.random.uniform(shape=(6, 3), minval=0, maxval=10, dtype=tf.float32),
    "segment_ids": tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32),
    "name": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    segment_ids: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([0, 0, 0, 0, 0, 0], dtype=tf.int32),
        tf.constant([0, 0, 0, 1, 1, 1], dtype=tf.int32),
        tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32),
        tf.constant([0, 1, 2, 2, 3, 3], dtype=tf.int32),
        tf.constant([0, 1, 2, 3, 4, 5], dtype=tf.int32)
    ])