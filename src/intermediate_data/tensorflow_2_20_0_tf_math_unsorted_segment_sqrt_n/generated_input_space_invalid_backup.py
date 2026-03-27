import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Union

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': tf.random.uniform(shape=(6, 4), dtype=tf.float32),
    'segment_ids': tf.constant([0, 1, 0, 2, 1, 0], dtype=tf.int32),
    'num_segments': 3,
    'name': None
}

# Task 2 & 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # segment_ids affects output shape through its rank
    segment_ids: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([0, 1, 0, 2, 1, 0], dtype=tf.int32),               # 1D, rank 1
        tf.constant([[0, 1], [1, 0]], dtype=tf.int32),                  # 2D, rank 2  
        tf.constant([[[0, 1]], [[1, 0]]], dtype=tf.int32),              # 3D, rank 3
        tf.constant([], dtype=tf.int32),                                # 0D scalar, rank 0
        tf.constant([[[[0, 1, 0]]]], dtype=tf.int32)                    # 4D, rank 4
    ])
    
    # num_segments affects output shape directly
    num_segments: List[Union[int, tf.Tensor]] = field(default_factory=lambda: [
        1,                                                              # Minimum value
        3,                                                              # Small value
        10,                                                             # Medium value  
        100,                                                            # Large value
        tf.constant(5)                                                  # Tensor input
    ])