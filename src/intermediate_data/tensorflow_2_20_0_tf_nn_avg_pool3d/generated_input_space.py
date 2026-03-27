import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, List

def call_func(inputs, ksize, strides, padding, data_format='NDHWC', name=None):
    return tf.nn.avg_pool3d(inputs[0], ksize, strides, padding, data_format, name)

random_input = tf.random.normal(shape=[2, 5, 7, 9, 4], dtype=tf.float32)
valid_test_case = {
    'inputs': [random_input],
    'ksize': [1, 2, 2, 2, 1],
    'strides': [1, 2, 2, 2, 1],
    'padding': 'VALID',
    'data_format': 'NDHWC',
}

@dataclass
class InputSpace:
    ksize: List[Union[int, List[int]]] = field(default_factory=lambda: [
        1,
        [1, 2, 2, 2, 1],
        [1, 3, 3, 3, 1],
        [2, 2, 2],
        [3, 3, 3]
    ])
    strides: List[Union[int, List[int]]] = field(default_factory=lambda: [
        1,
        [1, 2, 2, 2, 1],
        [1, 3, 3, 3, 1],
        [2, 2, 2],
        [3, 3, 3]
    ])
    padding: List[str] = field(default_factory=lambda: ['VALID', 'SAME'])
    data_format: List[str] = field(default_factory=lambda: ['NDHWC', 'NCDHW'])