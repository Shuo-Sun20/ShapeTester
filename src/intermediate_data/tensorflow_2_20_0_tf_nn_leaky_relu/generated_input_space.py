import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

# Original function
def call_func(inputs, alpha=0.2, name=None):
    return tf.nn.leaky_relu(features=inputs, alpha=alpha, name=name)

# Generate random tensor
random_tensor = tf.constant(np.random.randn(3, 4, 5).astype(np.float32))

# 1. Valid test case
valid_test_case = {
    "inputs": random_tensor,
    "alpha": 0.3,
    "name": None
}

# 2. Parameters affecting output shape: None (only "inputs" affects shape)

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # No parameters that affect output shape (excluding "inputs")
    pass