import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List

def call_func(inputs, stride, padding, data_format="NWC", dilations=1, name=None):
    input_tensor, filters_tensor = inputs[0], inputs[1]
    return tf.nn.conv1d(input=input_tensor, filters=filters_tensor, stride=stride, padding=padding, data_format=data_format, dilations=dilations, name=name)

# Task 1: valid_test_case
batch_size = 2
in_width = 10
in_channels = 3
filter_width = 4
out_channels = 5

input_tensor = tf.random.normal(shape=[batch_size, in_width, in_channels])
filters_tensor = tf.random.normal(shape=[filter_width, in_channels, out_channels])

valid_test_case = {
    "inputs": [input_tensor, filters_tensor],
    "stride": 1,
    "padding": "VALID",
    "data_format": "NWC",
    "dilations": 1,
    "name": None
}

# Task 4: InputSpace dataclass
@dataclass
class InputSpace:
    stride: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 2, 3, [2], [1,1,1]])
    padding: List[str] = field(default_factory=lambda: ["VALID", "SAME"])
    data_format: List[str] = field(default_factory=lambda: ["NWC", "NCW"])
    dilations: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 2, 3, [2], [1,1,1]])