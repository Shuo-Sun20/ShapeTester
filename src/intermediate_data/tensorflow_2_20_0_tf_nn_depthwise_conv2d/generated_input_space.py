import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Optional, Any

# 1. Define valid_test_case dictionary
batch_size = 2
height = 4
width = 4
in_channels = 3
channel_multiplier = 1

input_tensor = tf.convert_to_tensor(
    np.random.randn(batch_size, height, width, in_channels).astype(np.float32)
)
filter_tensor = tf.convert_to_tensor(
    np.random.randn(2, 2, in_channels, channel_multiplier).astype(np.float32)
)

valid_test_case = {
    'inputs': [input_tensor, filter_tensor],
    'strides': [1, 1, 1, 1],
    'padding': 'VALID',
    'data_format': None,
    'dilations': None,
    'name': None
}

# 2 & 3 & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 1, 2, 1],
        [1, 2, 1, 1],
        [1, 4, 4, 1]
    ])
    
    padding: List[Union[str, List[List[int]]]] = field(default_factory=lambda: [
        'VALID',
        'SAME',
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        [[0, 0], [2, 2], [2, 2], [0, 0]],
        [[0, 0], [3, 3], [3, 3], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]]
    ])
    
    data_format: List[Optional[str]] = field(default_factory=lambda: [
        None,
        'NHWC',
        'NCHW'
    ])
    
    dilations: List[Optional[List[int]]] = field(default_factory=lambda: [
        None,
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5]
    ])

# Example instantiation
var = InputSpace()