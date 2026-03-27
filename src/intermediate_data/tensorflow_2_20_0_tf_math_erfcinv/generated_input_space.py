import tensorflow as tf
from dataclasses import dataclass, field

valid_test_case = {
    'inputs': tf.constant([0., 0.5, 1., 1.5, 2.], dtype=tf.float32),
    'name': 'erfcinv_operation'
}

@dataclass
class InputSpace:
    # Only 'inputs' affects output shape, but it's excluded per instructions
    # Other parameters (like 'name') don't affect output shape
    # Therefore no fields are defined here
    pass