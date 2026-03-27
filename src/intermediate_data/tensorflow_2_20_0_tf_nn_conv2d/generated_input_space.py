import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [
        tf.random.normal(shape=[2, 32, 32, 3], dtype=tf.float32),
        tf.random.normal(shape=[3, 3, 3, 4], dtype=tf.float32)
    ],
    'strides': [1, 1, 1, 1],
    'padding': 'VALID',
    'data_format': 'NHWC',
    'dilations': [1, 1, 1, 1],
    'name': None
}

# 2-4. Define InputSpace 
@dataclass
class InputSpace:
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1]
    ])
    
    padding: List[Union[str, List[List[int]]]] = field(default_factory=lambda: [
        'VALID',
        'SAME',
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        [[0, 0], [2, 2], [2, 2], [0, 0]],
        [[0, 0], [3, 3], [3, 3], [0, 0]]
    ])
    
    data_format: List[str] = field(default_factory=lambda: [
        'NHWC',
        'NCHW'
    ])
    
    dilations: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1]
    ])