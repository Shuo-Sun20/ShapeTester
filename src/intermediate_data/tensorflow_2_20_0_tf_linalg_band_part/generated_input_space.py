import tensorflow as tf
from dataclasses import dataclass

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    "inputs": tf.random.normal(shape=(4, 4)),
    "num_lower": tf.constant(1, dtype=tf.int64),
    "num_upper": tf.constant(-1, dtype=tf.int64),
    "name": None  # Optional parameter
}

# Task 2: Identify parameters affecting output shape (excluding "inputs")
# - Only "inputs" parameter affects output shape, others only affect content values
# - Therefore, no additional parameters from call_func() affect output shape

# Task 3-4: Define InputSpace dataclass with all shape-affecting parameters
@dataclass
class InputSpace:
    # Since no parameters besides "inputs" affect output shape,
    # we only need to include "inputs" with its value ranges
    # However, "inputs" is excluded by the problem's instructions
    # Therefore, we define an empty dataclass as per requirements
    pass

# The class can be instantiated as:
var = InputSpace()