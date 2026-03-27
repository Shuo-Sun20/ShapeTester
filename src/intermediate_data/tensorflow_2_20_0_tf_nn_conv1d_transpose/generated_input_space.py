import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Union

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

# 2. & 3. & 4. Define InputSpace dataclass with shape-affecting parameters
@dataclass
class InputSpace:
    # output_shape: List[List[int]] - Would normally vary based on input, but here we treat as derived parameter
    strides: List[Union[int, List[int]]] = field(
        default_factory=lambda: [
            1, 2, 3, 4, 5,  # Discrete stride values
            [1], [2], [3]   # List form for strides
        ]
    )
    padding: List[str] = field(
        default_factory=lambda: ["VALID", "SAME"]
    )
    data_format: List[str] = field(
        default_factory=lambda: ["NWC", "NCW"]
    )
    dilations: List[Union[int, List[int]]] = field(
        default_factory=lambda: [
            1, 2, 3, 4, 5,  # Discrete dilation values
            [1], [2], [3]   # List form for dilations
        ]
    )

# Example instantiation
var = InputSpace()