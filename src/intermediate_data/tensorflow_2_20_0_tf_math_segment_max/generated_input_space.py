import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Any

# 1. Valid test case
data = tf.constant([[3, 7, 2], [9, 1, 5], [4, 8, 6]], dtype=tf.float32)
segment_ids = tf.constant([0, 0, 1], dtype=tf.int32)
valid_test_case = {
    "inputs": [data, segment_ids],
    "name": None
}

# 4. InputSpace definition
@dataclass
class InputSpace:
    inputs: List[List[Any]] = field(default_factory=lambda: [
        # Test case 1: Basic 2D case
        [tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32), 
         tf.constant([0, 1], dtype=tf.int32)],
        
        # Test case 2: 3D data
        [tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32), 
         tf.constant([0, 1], dtype=tf.int32)],
        
        # Test case 3: Single segment
        [tf.constant([1, 2, 3, 4], dtype=tf.float32), 
         tf.constant([0, 0, 0, 0], dtype=tf.int32)],
        
        # Test case 4: Multiple segments
        [tf.constant([[1], [2], [3], [4]], dtype=tf.float32), 
         tf.constant([0, 0, 1, 2], dtype=tf.int32)],
        
        # Test case 5: Empty data
        [tf.constant([], shape=[0, 2], dtype=tf.float32), 
         tf.constant([], dtype=tf.int32)]
    ])