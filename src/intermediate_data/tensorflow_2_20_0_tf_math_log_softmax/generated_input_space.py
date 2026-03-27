import tensorflow as tf
import numpy as np
from dataclasses import dataclass

# Task 1: Define valid_test_case dictionary
np.random.seed(42)
random_logits = tf.constant(np.random.randn(3, 4), dtype=tf.float32)
valid_test_case = {
    'inputs': random_logits,
    'axis': -1,
    'name': None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    axis: list = None
    
    def __post_init__(self):
        if self.axis is None:
            # For a 2D tensor with shape (3, 4), axis can be -2, -1, 0, 1
            # We'll include 5 typical values: two boundary values and common values
            self.axis = [-2, -1, 0, 1, -2]