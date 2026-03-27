import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    'inputs': tf.random.uniform(shape=[1, 4, 4, 3], minval=0, maxval=1, dtype=tf.float32),
    'block_size': 2,
    'data_format': 'NHWC',
    'name': None
}

@dataclass
class InputSpace:
    block_size: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    data_format: List[str] = field(default_factory=lambda: ["NHWC", "NCHW", "NCHW_VECT_C"])