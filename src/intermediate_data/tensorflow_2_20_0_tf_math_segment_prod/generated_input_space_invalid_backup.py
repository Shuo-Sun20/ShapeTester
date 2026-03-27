import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
    "segment_ids": tf.constant([0, 0, 1], dtype=tf.int32),
    "name": None
}

# Task 4: Define InputSpace class with discretized values (≤5 values per field)
@dataclass
class InputSpace:
    segment_ids: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([0, 0, 1, 2], dtype=tf.int32),       # 3 segments
        tf.constant([0, 1, 1], dtype=tf.int32),          # 2 segments
        tf.constant([0], dtype=tf.int32),                # 1 segment
        tf.constant([0, 0, 0], dtype=tf.int32),          # 1 segment, repeated
        tf.constant([0, 0, 1, 1, 2], dtype=tf.int32)     # 3 segments, varied lengths
    ])