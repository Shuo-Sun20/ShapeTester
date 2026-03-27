import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': [tf.random.normal([2, 5, 5, 3]), tf.random.normal([3, 3, 3])],
    'strides': [1, 1, 1, 1],
    'padding': 'SAME',
    'data_format': 'NHWC',
    'dilations': [1, 1, 1, 1],
    'name': None
}

@dataclass
class InputSpace:
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 1, 2, 1],
        [1, 2, 1, 1]
    ])
    padding: List[str] = field(default_factory=lambda: ['SAME', 'VALID'])
    dilations: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 1, 2, 1],
        [1, 2, 1, 1]
    ])