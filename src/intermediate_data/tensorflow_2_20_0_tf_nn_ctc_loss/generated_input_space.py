import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    "inputs": [
        tf.constant([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=tf.int64),
        tf.random.uniform([12, 3, 6], dtype=tf.float32)
    ],
    "label_length": tf.constant([3, 2, 5], dtype=tf.int32),
    "logit_length": tf.constant([12, 12, 12], dtype=tf.int32),
    "logits_time_major": True,
    "unique": None,
    "blank_index": 0,
    "name": "ctc_loss_dense"
}

# Task 2-4: Define InputSpace class with parameters affecting output shape
@dataclass
class InputSpace:
    label_length: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([1], dtype=tf.int32),
        tf.constant([2, 3], dtype=tf.int32),
        tf.constant([4, 5, 6], dtype=tf.int32),
        tf.constant([7, 8, 9, 10], dtype=tf.int32),
        tf.constant([11, 12, 13, 14, 15], dtype=tf.int32)
    ])
    logit_length: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([5], dtype=tf.int32),
        tf.constant([10, 15], dtype=tf.int32),
        tf.constant([20, 25, 30], dtype=tf.int32),
        tf.constant([35, 40, 45, 50], dtype=tf.int32),
        tf.constant([55, 60, 65, 70, 75], dtype=tf.int32)
    ])