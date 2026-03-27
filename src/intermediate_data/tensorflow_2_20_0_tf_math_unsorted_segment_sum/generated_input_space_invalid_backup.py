import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Valid test case dictionary
valid_test_case = {
    "inputs": [tf.constant(np.random.randn(3, 4).astype(np.float32))],
    "segment_ids": tf.constant([0, 1, 0], dtype=tf.int32),
    "num_segments": tf.constant(2, dtype=tf.int32),
    "name": None
}

# 2. Parameters affecting output shape: segment_ids, num_segments

# 3 & 4. InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    segment_ids: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([0, 1, 0], dtype=tf.int32),
        tf.constant([0, 0, 1, 1], dtype=tf.int32),
        tf.constant([[0, 1], [1, 0]], dtype=tf.int32),
        tf.constant([-1, 0, 1], dtype=tf.int32),
        tf.constant([2, 2, 2], dtype=tf.int32)
    ])
    
    num_segments: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant(0, dtype=tf.int32),
        tf.constant(1, dtype=tf.int32),
        tf.constant(2, dtype=tf.int32),
        tf.constant(3, dtype=tf.int32),
        tf.constant(5, dtype=tf.int32)
    ])