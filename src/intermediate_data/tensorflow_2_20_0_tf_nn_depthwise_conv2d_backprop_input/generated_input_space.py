import tensorflow as tf
from dataclasses import dataclass
from typing import List, Union

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [
        tf.constant([2, 4, 4, 3], dtype=tf.int32),  # input_sizes
        tf.random.normal([3, 3, 3, 2], dtype=tf.float32),  # filter_tensor
        tf.random.normal([2, 4, 4, 6], dtype=tf.float32)   # out_backprop
    ],
    'strides': [1, 1, 1, 1],
    'padding': "SAME",
    'data_format': 'NHWC',
    'dilations': [1, 1, 1, 1],
}

@dataclass
class InputSpace:
    # Task 4: Define parameters affecting output shape with discretized value ranges
    strides: List[List[int]] = None
    padding: List[Union[str, List[List[int]]]] = None
    data_format: List[str] = None
    dilations: List[List[int]] = None
    
    def __post_init__(self):
        if self.strides is None:
            # Task 3: Discretized strides (height/width dimensions only)
            self.strides = [
                [1, 1, 1, 1],    # no stride
                [1, 2, 2, 1],    # stride 2
                [1, 3, 3, 1],    # stride 3
                [1, 4, 4, 1],    # stride 4
                [1, 5, 5, 1],    # stride 5
            ]
        
        if self.padding is None:
            # Task 3: Discretized padding values (limited to 5)
            self.padding = [
                "SAME",                     # same padding
                "VALID",                    # valid padding
                [[0,0], [1,1], [1,1], [0,0]],  # explicit padding for NHWC
                [[0,0], [0,0], [1,1], [1,1]],  # explicit padding for NCHW
                [[0,0], [2,2], [2,2], [0,0]],  # larger explicit padding
            ]
        
        if self.data_format is None:
            # Task 3: All possible data_format values (only 2 values)
            self.data_format = ["NHWC", "NCHW"]
        
        if self.dilations is None:
            # Task 3: Discretized dilations (height/width dimensions only)
            self.dilations = [
                [1, 1, 1, 1],    # no dilation
                [1, 2, 2, 1],    # dilation 2
                [1, 3, 3, 1],    # dilation 3
                [1, 4, 4, 1],    # dilation 4
                [1, 5, 5, 1],    # dilation 5
            ]