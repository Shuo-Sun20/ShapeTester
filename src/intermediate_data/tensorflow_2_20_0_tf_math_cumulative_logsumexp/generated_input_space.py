import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, axis=0, exclusive=False, reverse=False, name=None):
    return tf.math.cumulative_logsumexp(x=inputs, axis=axis, exclusive=exclusive, reverse=reverse, name=name)

example_input = tf.constant(np.random.randn(3, 4).astype(np.float32))
example_output = call_func(inputs=example_input, axis=1, exclusive=True)

valid_test_case = {
    "inputs": example_input,
    "axis": 1,
    "exclusive": True,
    "reverse": False,
    "name": None
}

@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [0, 1, -1, -2, 2])
    exclusive: list = field(default_factory=lambda: [True, False])
    reverse: list = field(default_factory=lambda: [True, False])