import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Valid test case for call_func()
valid_test_case = {
    "inputs": tf.constant(tf.random.normal(shape=(1, 4, 4, 1), dtype=tf.float32)),
    "ksize": 2,
    "strides": 2,
    "padding": "SAME",
    "data_format": None,
    "name": None
}

# Parameters that affect output shape: ksize, strides, padding, data_format
@dataclass
class InputSpace:
    # ksize: int or list of ints, affects pooling window size
    ksize: List[Union[int, List[int]]] = field(default_factory=lambda: [
        1,
        2,
        3,
        [1, 2],
        [2, 2]
    ])
    
    # strides: int or list of ints, affects sliding window stride
    strides: List[Union[int, List[int]]] = field(default_factory=lambda: [
        1,
        2,
        3,
        [1, 1],
        [2, 2]
    ])
    
    # padding: string or explicit padding list, affects padding method
    padding: List[Union[str, List[List[int]]]] = field(default_factory=lambda: [
        "VALID",
        "SAME",
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        [[0, 0], [2, 2], [2, 2], [0, 0]]
    ])
    
    # data_format: string, affects channel dimension order
    data_format: List[Optional[str]] = field(default_factory=lambda: [
        None,
        "NHWC",
        "NCHW"
    ])