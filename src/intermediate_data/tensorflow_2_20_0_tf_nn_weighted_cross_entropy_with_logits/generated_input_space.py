import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Provided code
def call_func(inputs, pos_weight, name=None):
    labels = inputs[0]
    logits = inputs[1]
    return tf.nn.weighted_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        pos_weight=pos_weight,
        name=name
    )

np.random.seed(42)
batch_size = 4
logits = tf.constant(np.random.randn(batch_size).astype(np.float32))
labels = tf.constant(np.random.uniform(0, 1, batch_size).astype(np.float32))
pos_weight = tf.constant(1.5, dtype=tf.float32)

valid_test_case = {
    'inputs': [labels, logits],
    'pos_weight': pos_weight
}

@dataclass
class InputSpace:
    # No parameters from call_func() affect output shape except 'inputs' 
    # (which is excluded by the task requirements)
    pass