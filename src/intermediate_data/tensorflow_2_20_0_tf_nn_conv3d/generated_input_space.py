import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [tf.constant(np.random.randn(2, 10, 10, 10, 4).astype(np.float32)),
               tf.constant(np.random.randn(3, 3, 3, 4, 6).astype(np.float32))],
    'strides': [1, 2, 2, 2, 1],
    'padding': 'SAME',
    'data_format': 'NDHWC',
    'dilations': [1, 1, 1, 1, 1],
    'name': 'example_conv3d'
}

# 2, 3 & 4.
@dataclass
class InputSpace:
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 3, 3, 3, 1],
        [1, 1, 2, 3, 1],
        [1, 3, 2, 1, 1]
    ])
    padding: List[str] = field(default_factory=lambda: ['SAME', 'VALID'])
    data_format: List[str] = field(default_factory=lambda: ['NDHWC', 'NCDHW'])
    dilations: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 3, 3, 3, 1],
        [1, 1, 2, 3, 1],
        [1, 3, 2, 1, 1]
    ])