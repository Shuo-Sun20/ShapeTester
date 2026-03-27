import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [
        tf.constant(np.random.randn(2, 5, 5, 3).astype(np.float32)),
        tf.constant(np.random.randn(3, 3, 3).astype(np.float32))
    ],
    'strides': [1, 1, 1, 1],
    'padding': 'SAME',
    'data_format': 'NHWC',
    'dilations': [1, 1, 1, 1],
    'name': None
}

# Task 4: Define InputSpace class
@dataclass
class InputSpace:
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1]
    ])
    
    padding: List[str] = field(default_factory=lambda: [
        'SAME',
        'VALID'
    ])
    
    dilations: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1]
    ])