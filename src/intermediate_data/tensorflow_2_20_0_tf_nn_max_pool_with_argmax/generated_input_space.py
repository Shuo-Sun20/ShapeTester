import tensorflow as tf
from dataclasses import dataclass, field

valid_test_case = {
    'inputs': tf.random.normal(shape=[2, 8, 8, 3]),
    'ksize': [1, 2, 2, 1],
    'strides': [1, 2, 2, 1],
    'padding': 'VALID',
    'data_format': 'NHWC',
    'output_dtype': tf.int64,
    'include_batch_in_index': False,
    'name': None
}

@dataclass
class InputSpace:
    ksize: list = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1]
    ])
    
    strides: list = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 5, 5, 1]
    ])
    
    padding: list = field(default_factory=lambda: ['VALID', 'SAME'])
    
    data_format: list = field(default_factory=lambda: ['NHWC'])