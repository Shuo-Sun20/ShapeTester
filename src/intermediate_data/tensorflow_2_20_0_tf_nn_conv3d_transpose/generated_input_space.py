import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Tuple

# 1. valid_test_case definition
valid_test_case = {
    'inputs': [
        tf.constant(np.random.randn(2, 5, 5, 5, 3), dtype=tf.float32),  # input_tensor
        tf.constant(np.random.randn(3, 3, 3, 4, 3), dtype=tf.float32),  # filters_tensor
        tf.constant([2, 10, 10, 10, 4], dtype=tf.int32)                 # output_shape_tensor
    ],
    'strides': [1, 2, 2, 2, 1],
    'padding': 'SAME',
    'data_format': 'NDHWC',
    'dilations': 1
}

# 2. & 3. & 4. InputSpace dataclass
@dataclass
class InputSpace:
    strides: List[Union[int, List[int]]] = field(
        default_factory=lambda: [
            1,
            2,
            [1, 2, 2, 2, 1],
            [1, 1, 2, 2, 1],
            [1, 2, 1, 2, 1]
        ]
    )
    padding: List[str] = field(
        default_factory=lambda: ['VALID', 'SAME']
    )
    data_format: List[str] = field(
        default_factory=lambda: ['NDHWC', 'NCDHW']
    )
    dilations: List[Union[int, List[int]]] = field(
        default_factory=lambda: [
            1,
            2,
            [1, 2, 2, 2, 1],
            [1, 1, 2, 2, 1],
            [1, 2, 1, 2, 1]
        ]
    )