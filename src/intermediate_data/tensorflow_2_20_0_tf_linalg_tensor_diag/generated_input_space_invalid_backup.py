import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List

# Task 1: Valid test case
valid_test_case = {
    'inputs': tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32),
    'name': None
}

# Task 2, 3, 4: InputSpace class
@dataclass
class InputSpace:
    # The only parameter affecting output shape is 'inputs' (diagonal tensor)
    # name parameter doesn't affect output shape
    # For diagonal: we consider different rank-1 tensor shapes
    inputs: List[tf.Tensor] = None
    
    def __post_init__(self):
        if self.inputs is None:
            # Create 5 different rank-1 tensors with varying sizes
            # Allowed dtypes: bfloat16, half, float32, float64, int32, int64, complex64, complex128
            # We'll use float32 for simplicity
            self.inputs = [
                tf.constant([], dtype=tf.float32),  # empty tensor
                tf.constant([1.0], dtype=tf.float32),  # size 1
                tf.constant([1.0, 2.0], dtype=tf.float32),  # size 2
                tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32),  # size 4
                tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32),  # size 5
            ]