import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

# Generate random input tensor
np.random.seed(42)
random_input = tf.constant(np.random.randn(3, 4, 5).astype(np.float32))

# 1. valid_test_case dictionary
valid_test_case = {
    'inputs': random_input,
    'axis': [0, 2],
    'keepdims': True,
    'name': "example_reduce_euclidean_norm"
}

# 2 & 3. Parameters affecting output shape and their value spaces:
@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [None, 0, [0, 1], [0, 2], [-1]])
    keepdims: list = field(default_factory=lambda: [True, False])