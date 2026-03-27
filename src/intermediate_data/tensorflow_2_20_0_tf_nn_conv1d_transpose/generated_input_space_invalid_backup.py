import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case dict
valid_test_case = {
    'inputs': [tf.random.normal([2, 5, 3]), tf.random.normal([3, 4, 3])],
    'output_shape': tf.constant([2, 10, 4]),
    'strides': 2,
    'padding': "SAME",
    'data_format': "NWC",
    'dilations': 1,
    'name': None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    output_shape: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([2, 5, 4]),     # Minimum valid width (stride=1, SAME)
        tf.constant([2, 10, 4]),    # Example case
        tf.constant([2, 15, 4]),    # Intermediate width
        tf.constant([2, 20, 4]),    # Large width
        tf.constant([2, 25, 4])     # Very large width
    ])
    strides: List[int] = field(default_factory=lambda: [
        1, 2, 3, 4, 5               # stride values from 1 to 5
    ])
    padding: List[str] = field(default_factory=lambda: [
        "VALID", "SAME"
    ])
    data_format: List[str] = field(default_factory=lambda: [
        "NWC", "NCW"
    ])
    dilations: List[int] = field(default_factory=lambda: [
        1, 2, 3, 4, 5               # dilation values from 1 to 5
    ])