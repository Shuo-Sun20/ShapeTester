import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional

# Original function definition from the problem
def call_func(inputs, name=None):
    a, x = inputs[0], inputs[1]
    return tf.math.igammac(a=a, x=x, name=name)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        tf.constant([[1.5, 2.5], [3.5, 4.5]], dtype=tf.float32),  # a tensor
        tf.constant([[0.5, 1.5], [2.5, 3.5]], dtype=tf.float32)   # x tensor
    ],
    "name": "test_igammac"
}

# Task 2: Parameters affecting output shape (excluding 'inputs') is only 'name'
# Task 3-4: Define InputSpace dataclass with discretized parameter values
@dataclass
class InputSpace:
    # 'name' is a string parameter that can be None or a custom string
    # It doesn't affect the output tensor shape, but it's included per requirements
    name: list[Optional[str]] = field(
        default_factory=lambda: [None, "op1", "custom_name", "igammac_test", "final_output"]
    )