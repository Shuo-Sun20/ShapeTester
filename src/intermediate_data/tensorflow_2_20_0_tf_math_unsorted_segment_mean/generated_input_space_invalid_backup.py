import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [tf.constant(np.random.randn(6, 3, 4).astype(np.float32))],
    "segment_ids": tf.constant([0, 1, 0, 2, 1, 2], dtype=tf.int32),
    "num_segments": 4,
    "name": None
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters affecting the output shape of call_func.
    """
    segment_ids: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([0, 1, 0, 2, 1, 2], dtype=tf.int32),
        tf.constant([0, 0, 0, 0], dtype=tf.int32),
        tf.constant([0, 1, 2, 3, 4], dtype=tf.int32),
        tf.constant([0], dtype=tf.int32),
        tf.constant([1, 1, 1, 0, 0, 2, 2], dtype=tf.int32)
    ])
    
    num_segments: List[int] = field(default_factory=lambda: [
        0,  # Boundary: minimum valid value
        1,  # Typical: single segment
        3,  # Typical: matches segment_ids range
        5,  # Typical: larger than segment_ids range
        10  # Boundary: larger than typical range
    ])