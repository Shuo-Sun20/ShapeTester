import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [tf.constant(np.random.randn(2, 6, 8, 8, 3).astype(np.float32))],
    'ksize': [1, 2, 2, 2, 1],
    'strides': [1, 2, 2, 2, 1],
    'padding': 'VALID',
    'data_format': 'NDHWC',
    'name': None
}

# 2. & 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    ksize: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 3, 3, 3, 1],
        [1, 4, 4, 4, 1],
        [1, 2, 3, 4, 1]
    ])
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 3, 3, 3, 1],
        [1, 4, 4, 4, 1],
        [1, 2, 3, 4, 1]
    ])
    padding: List[str] = field(default_factory=lambda: ['VALID', 'SAME'])
    data_format: List[str] = field(default_factory=lambda: ['NDHWC', 'NCDHW'])