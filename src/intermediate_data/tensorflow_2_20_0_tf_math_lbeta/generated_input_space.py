import tensorflow as tf
from dataclasses import dataclass

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': tf.random.uniform(shape=[2, 3, 4], minval=0.1, maxval=5.0, dtype=tf.float32)
}

# 2, 3, 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # There are no parameters besides "inputs" that affect output shape
    # As per analysis, the output shape is always input shape without last dimension
    # and no other call_func parameters exist to modify this behavior
    pass