import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    "inputs": tf.constant([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]], dtype=tf.int32),
    "segment_ids": tf.constant([0, 0, 1], dtype=tf.int32),
    "name": None
}

# 2. Parameters affecting output shape (excluding inputs):
# Only segment_ids affects output shape by determining the number of segments

@dataclass
class InputSpace:
    segment_ids: List[Union[List[int], tf.Tensor]] = field(
        default_factory=lambda: [
            # Boundary case: 0 segments (empty)
            tf.constant([], dtype=tf.int32),
            # Small number of segments
            tf.constant([0, 0, 0], dtype=tf.int32),
            # Medium number of segments
            tf.constant([0, 0, 1, 1, 2, 2], dtype=tf.int32),
            # Gaps in segment IDs
            tf.constant([0, 0, 2, 2, 4, 4], dtype=tf.int32),
            # Maximum segment ID = 0 (all same segment)
            tf.constant([0, 0, 0, 0], dtype=tf.int32)
        ]
    )