import tensorflow as tf
from dataclasses import dataclass

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.constant([-3.0, -1.0, 0.0, 6.0, 10.0], dtype=tf.float32),
    'name': None
}

# 2. Parameters affecting output shape (excluding 'inputs'):
# None. Only 'inputs' affects shape, 'name' only provides metadata.

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    pass  # No parameters affecting shape (excluding 'inputs')