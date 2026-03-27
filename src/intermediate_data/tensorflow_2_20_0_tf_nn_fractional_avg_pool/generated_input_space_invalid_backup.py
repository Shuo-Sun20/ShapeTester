import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None):
    return tf.nn.fractional_avg_pool(value=inputs, pooling_ratio=pooling_ratio, pseudo_random=pseudo_random, overlapping=overlapping, seed=seed, name=name)[0]

# 1. Define valid_test_case
example_input = tf.constant(np.random.randn(4, 10, 10, 3).astype(np.float32))
valid_test_case = {
    "inputs": example_input,
    "pooling_ratio": [1.0, 1.44, 1.73, 1.0],
    "pseudo_random": False,
    "overlapping": False,
    "seed": 0,
    "name": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    pooling_ratio: list = field(default_factory=lambda: [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.25, 1.25, 1.0],
        [1.0, 1.5, 1.5, 1.0],
        [1.0, 1.75, 1.75, 1.0],
        [1.0, 2.0, 2.0, 1.0]
    ])
    overlapping: list = field(default_factory=lambda: [True, False])

# Example instantiation
var = InputSpace()