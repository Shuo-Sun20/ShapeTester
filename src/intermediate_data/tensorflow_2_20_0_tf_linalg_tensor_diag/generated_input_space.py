import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Task 1: Valid test case
valid_test_case = {
    'inputs': tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32),
    'name': None
}

# Task 2 & 3: Identify parameters affecting output shape and their value spaces
# The only parameter that affects output shape is 'inputs', and we're excluding it.
# The 'name' parameter is for operation naming only and doesn't affect output shape.
# Therefore, there are no additional parameters affecting output shape.

# Task 4: InputSpace class
@dataclass
class InputSpace:
    # No parameters affecting output shape besides 'inputs' (which we're excluding)
    pass