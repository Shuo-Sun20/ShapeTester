import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.random.normal(shape=[5, 2, 6]),
    'sequence_length': tf.constant([3, 5], dtype=tf.int32),
    'merge_repeated': True,
    'blank_index': None
}

# 2 & 3 & 4. Define InputSpace class with parameters affecting output shape
@dataclass
class InputSpace:
    """
    Contains all parameters affecting output tensor shape for CTC greedy decoder.
    """
    sequence_length: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([0, 0], dtype=tf.int32),  # min length
        tf.constant([2, 2], dtype=tf.int32),
        tf.constant([3, 5], dtype=tf.int32),
        tf.constant([5, 3], dtype=tf.int32),
        tf.constant([5, 5], dtype=tf.int32)   # max length (matches max_time=5)
    ])
    
    merge_repeated: List[bool] = field(default_factory=lambda: [True, False])
    
    blank_index: List[Optional[int]] = field(default_factory=lambda: [
        None,        # default (num_classes-1 = 5)
        0,           # first class
        2,           # middle class
        4,           # near end class
        -1           # maps to 5 (same as default)
    ])