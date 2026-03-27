import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    "inputs": [
        tf.constant([1, 2, 3, 1, 2, 3, 4], dtype=tf.int32)
    ],
    "minlength": None,
    "maxlength": None,
    "dtype": tf.int32,
    "name": None,
    "axis": None,
    "binary_output": False
}

@dataclass
class InputSpace:
    minlength: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 5, 10])
    maxlength: List[Optional[int]] = field(default_factory=lambda: [None, 1, 4, 8, 100])
    axis: List[Optional[int]] = field(default_factory=lambda: [None, 0, -1])