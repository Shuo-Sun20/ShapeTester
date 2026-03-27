import tensorflow as tf
from dataclasses import dataclass
from typing import List, Union, Optional

# 1. Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=[2, 3, 4], dtype=tf.float32),
    "axes": [0, 1],
    "frequency_weights": tf.abs(tf.random.normal(shape=[2, 3, 4], dtype=tf.float32)) + 0.1,
    "keepdims": False,
    "name": None
}

# 3-4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters affecting output shape (except inputs)
    axes: List[List[int]] = None
    keepdims: List[bool] = None
    
    def __post_init__(self):
        if self.axes is None:
            # For a 3D input tensor of shape [2, 3, 4], discretized axes combinations
            self.axes = [
                [0],       # Reduce dimension 0
                [1],       # Reduce dimension 1  
                [2],       # Reduce dimension 2
                [0, 1],    # Reduce dimensions 0 and 1
                [1, 2]     # Reduce dimensions 1 and 2
            ]
        if self.keepdims is None:
            self.keepdims = [True, False]