import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Tuple

# 1. Define valid_test_case dictionary
batch_size = 2
height = 4
width = 4
in_channels = 3
out_channels = 5
kernel_h = 3
kernel_w = 3

input_tensor = tf.random.normal(shape=[batch_size, height, width, in_channels])
filters_tensor = tf.random.normal(shape=[kernel_h, kernel_w, out_channels, in_channels])

output_h = height * 2
output_w = width * 2
output_shape = tf.constant([batch_size, output_h, output_w, out_channels], dtype=tf.int32)

strides = [1, 2, 2, 1]
padding = 'SAME'
data_format = 'NHWC'
dilations = None
name = None

valid_test_case = {
    'inputs': [input_tensor, filters_tensor],
    'output_shape': output_shape,
    'strides': strides,
    'padding': padding,
    'data_format': data_format,
    'dilations': dilations,
    'name': name
}

# 2. Parameters affecting output shape (excluding "inputs"):
# - output_shape, strides, padding, data_format, dilations

@dataclass
class InputSpace:
    # Parameters affecting output shape with discretized value spaces
    output_shape: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([2, 8, 8, 5], dtype=tf.int32),      # stride 2, SAME
        tf.constant([2, 7, 7, 5], dtype=tf.int32),      # stride 2, VALID
        tf.constant([2, 12, 12, 5], dtype=tf.int32),    # stride 3, SAME
        tf.constant([2, 16, 16, 5], dtype=tf.int32),    # stride 4, SAME
        tf.constant([2, 4, 4, 5], dtype=tf.int32)       # stride 1, SAME
    ])
    
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1],
        [1, 1, 2, 1]
    ])
    
    padding: List[Union[str, List[List[int]]]] = field(default_factory=lambda: [
        'SAME',
        'VALID',
        [[0, 0], [1, 1], [1, 1], [0, 0]],  # NHWC explicit padding
        [[0, 0], [0, 0], [1, 1], [1, 1]],  # NCHW explicit padding
        [[0, 0], [0, 0], [0, 0], [0, 0]]   # no padding
    ])
    
    data_format: List[str] = field(default_factory=lambda: [
        'NHWC',
        'NCHW'
    ])
    
    dilations: List[Union[None, List[int]]] = field(default_factory=lambda: [
        None,
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 4, 1]
    ])

# The InputSpace can be successfully instantiated:
var = InputSpace()