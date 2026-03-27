import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Union, Optional

# 1. Define valid_test_case based on the example provided
batch_size = 2
in_height = 5
in_width = 5
in_channels = 3
channel_multiplier = 2
out_channels = 4
filter_height = 3
filter_width = 3

input_tensor = tf.random.normal([batch_size, in_height, in_width, in_channels])
depthwise_filter = tf.random.normal([filter_height, filter_width, in_channels, channel_multiplier])
pointwise_filter = tf.random.normal([1, 1, channel_multiplier * in_channels, out_channels])

valid_test_case = {
    'inputs': [input_tensor, depthwise_filter, pointwise_filter],
    'strides': [1, 1, 1, 1],
    'padding': 'SAME',
    'data_format': 'NHWC',
    'dilations': None,
    'name': 'separable_conv2d_example'
}

# 2,3,4. Define InputSpace class
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
        'SAME',
        'VALID',
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        [[0, 0], [2, 2], [2, 2], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [0, 0]]
    ])
    data_format: List[str] = field(default_factory=lambda: [
        'NHWC',
        'NCHW'
    ])
    dilations: List[Optional[List[int]]] = field(default_factory=lambda: [
        None,
        [1, 1],
        [2, 2],
        [3, 3],
        [1, 2]
    ])