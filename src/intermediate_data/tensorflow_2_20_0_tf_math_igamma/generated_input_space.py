import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(inputs, name=None):
    a, x = inputs
    return tf.math.igamma(a, x, name)

valid_test_case = {
    'inputs': [
        tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
        tf.constant([0.5, 1.5, 2.5], dtype=tf.float32)
    ],
    'name': None
}

@dataclass
class InputSpace:
    name: List[Optional[str]] = field(default_factory=lambda: [None, 'test_op', 'gamma_op', 'custom_name', ''])