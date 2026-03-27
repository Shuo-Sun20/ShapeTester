import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# 1. valid_test_case dictionary
valid_test_case = {
    'weights': tf.random.normal([10000, 128]),
    'biases': tf.random.normal([10000]),
    'labels': tf.random.uniform([32, 2], 0, 10000, dtype=tf.int64),
    'inputs': tf.random.normal([32, 128]),
    'num_sampled': 50,
    'num_classes': 10000,
    'num_true': 2,
    'sampled_values': None,
    'remove_accidental_hits': False,
    'name': None
}

# 4. InputSpace class
@dataclass
class InputSpace:
    labels: List[tf.Tensor] = field(default_factory=lambda: [
        tf.random.uniform([1, 1], 0, 10000, dtype=tf.int64),
        tf.random.uniform([1, 2], 0, 10000, dtype=tf.int64),
        tf.random.uniform([16, 1], 0, 10000, dtype=tf.int64),
        tf.random.uniform([16, 2], 0, 10000, dtype=tf.int64),
        tf.random.uniform([32, 2], 0, 10000, dtype=tf.int64)
    ])