import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List

# Define the call_func as provided
def call_func(inputs, rate, padding, name=None):
    value, filters = inputs[0], inputs[1]
    output = tf.nn.atrous_conv2d(value=value, filters=filters, rate=rate, padding=padding, name=name)
    return output

# Generate random input tensors as in the example
batch_size = 2
in_height = 32
in_width = 32
in_channels = 3
out_channels = 16
filter_height = 3
filter_width = 3
rate = 2

value = tf.constant(
    np.random.randn(batch_size, in_height, in_width, in_channels).astype(np.float32)
)
filters = tf.constant(
    np.random.randn(filter_height, filter_width, in_channels, out_channels).astype(np.float32)
)

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    'inputs': [value, filters],
    'rate': rate,
    'padding': 'SAME'
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    rate: List[int] = None
    padding: List[str] = None
    
    def __post_init__(self):
        # Task 3: Define discretized value spaces
        # For rate (positive int32): including boundary values and 5 typical values
        if self.rate is None:
            self.rate = [1, 2, 3, 4, 5]  # max 5 values
        # For padding (discrete parameter): all possible values
        if self.padding is None:
            self.padding = ['VALID', 'SAME']