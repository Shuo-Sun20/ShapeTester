import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=[1, 2, 2, 4]),
    "block_size": 2,
    "data_format": "NHWC",
    "name": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    block_size: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    data_format: List[str] = field(default_factory=lambda: ["NHWC", "NCHW", "NCHW_VECT_C"])