import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional

def call_func(inputs, padding, strides=None, dilations=None, name=None, data_format=None):
    input_tensor, filters_tensor = inputs
    return tf.nn.convolution(
        input=input_tensor,
        filters=filters_tensor,
        padding=padding,
        strides=strides,
        dilations=dilations,
        name=name,
        data_format=data_format
    )

# Construct valid input tensors
batch_size = 2
input_spatial_shape = [5, 5]
in_channels = 3
out_channels = 4
spatial_filter_shape = [3, 3]

input_tensor = tf.random.normal(shape=[batch_size] + input_spatial_shape + [in_channels])
filters_tensor = tf.random.normal(shape=spatial_filter_shape + [in_channels, out_channels])

valid_test_case = {
    "inputs": [input_tensor, filters_tensor],
    "padding": "SAME",
    "strides": [1, 1],
    "dilations": [1, 1],
    "data_format": None
}

@dataclass
class InputSpace:
    padding: List[str] = field(default_factory=lambda: ["VALID", "SAME"])
    strides: List[List[int]] = field(default_factory=lambda: [[1, 1], [1, 2], [2, 1], [2, 2], [3, 3]])
    dilations: List[List[int]] = field(default_factory=lambda: [[1, 1], [1, 2], [2, 1], [2, 2], [3, 3]])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "NHWC", "NCHW"])