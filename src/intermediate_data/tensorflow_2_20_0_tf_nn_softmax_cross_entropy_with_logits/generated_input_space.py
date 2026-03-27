import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
logits = tf.random.uniform(shape=(2, 3), minval=-1, maxval=1, dtype=tf.float32)
labels = tf.random.uniform(shape=(2, 3), minval=0, maxval=1, dtype=tf.float32)
labels = labels / tf.reduce_sum(labels, axis=1, keepdims=True)
valid_test_case = {
    "inputs": [labels, logits],
    "axis": -1,
    "name": None
}

# 2 & 3. Parameters affecting output shape and their value space

@dataclass
class InputSpace:
    axis: List[int] = field(default_factory=lambda: [-3, -2, -1, 0, 1])