import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union

def call_func(inputs, ksize, strides, padding, data_format='NHWC', name=None):
    return tf.nn.avg_pool2d(input=inputs, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)

np.random.seed(42)
example_input = tf.constant(np.random.randn(2, 8, 8, 3).astype(np.float32))
valid_test_case = {
    'inputs': example_input,
    'ksize': [1, 2, 2, 1],
    'strides': [1, 2, 2, 1],
    'padding': 'VALID',
    'data_format': 'NHWC',
    'name': None
}

@dataclass
class InputSpace:
    ksize: List[List[int]] = field(default_factory=lambda: [
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1],
        [1, 6, 6, 1]
    ])
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1]
    ])
    padding: List[str] = field(default_factory=lambda: ['VALID', 'SAME'])
    data_format: List[str] = field(default_factory=lambda: ['NHWC', 'NCHW'])