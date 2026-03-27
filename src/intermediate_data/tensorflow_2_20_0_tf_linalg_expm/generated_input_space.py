import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, name=None):
    return tf.linalg.expm(inputs, name=name)

# 1. Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=(2, 3, 3), dtype=tf.float32),
    "name": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    name: List[Optional[str]] = field(
        default_factory=lambda: [None, "expm_op", "matrix_exp", "test_op", "my_expm"]
    )