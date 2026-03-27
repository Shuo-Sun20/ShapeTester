import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List

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
    "name": None
}

# Task 2 & 3: Parameters affecting output shape and their value spaces
# The only parameter besides "inputs" is "name"
# "name" is a string parameter that doesn't affect output shape but affects operation name
# It's a discrete parameter with possible values being strings or None

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # name doesn't affect output shape but is required by the feedback
    name: List[Optional[str]] = field(default_factory=lambda: [
        None,                     # Default name (no name)
        "igammac_op",            # Typical operation name
        "upper_gamma_func",      # Descriptive name
        "gamma_q",               # Alternative mathematical name
        "my_igammac",            # Custom operation name
        ""                       # Empty string name
    ])