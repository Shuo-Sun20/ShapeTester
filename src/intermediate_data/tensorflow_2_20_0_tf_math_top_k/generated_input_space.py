import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, k, sorted=True, index_type=tf.int32, name=None):
    return tf.math.top_k(input=inputs, k=k, sorted=sorted, index_type=index_type, name=name)

# Generate random input tensor
input_tensor = tf.constant(np.random.randn(3, 4, 5), dtype=tf.float32)
k_value = 2

# 1. Define valid_test_case
valid_test_case = {
    "inputs": input_tensor,
    "k": k_value,
    "sorted": True,
    "index_type": tf.int32,
    "name": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    k: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])