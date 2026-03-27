import tensorflow as tf
from dataclasses import dataclass
from typing import List

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], dtype=tf.int32),
    'segment_ids': tf.constant([0, 1, 0], dtype=tf.int32),
    'num_segments': tf.constant(2, dtype=tf.int32),
    'name': None
}

# 2. Parameters affecting output shape (except inputs): segment_ids, num_segments

# 3-4. Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    segment_ids: List[tf.Tensor] = None
    num_segments: List[tf.Tensor] = None
    
    def __post_init__(self):
        if self.segment_ids is None:
            self.segment_ids = [
                tf.constant([0], dtype=tf.int32),  # scalar
                tf.constant([0, 1, 0], dtype=tf.int32),  # 1D with 3 elements
                tf.constant([[0, 1], [1, 0]], dtype=tf.int32),  # 2x2
                tf.constant([0, 1, 2, 0, 1], dtype=tf.int32),  # 1D with 5 elements
                tf.constant([], dtype=tf.int32)  # empty
            ]
        if self.num_segments is None:
            self.num_segments = [
                tf.constant(0, dtype=tf.int32),  # boundary
                tf.constant(1, dtype=tf.int32),
                tf.constant(2, dtype=tf.int32),
                tf.constant(5, dtype=tf.int32),
                tf.constant(10, dtype=tf.int32)
            ]

# Example instantiation
var = InputSpace()