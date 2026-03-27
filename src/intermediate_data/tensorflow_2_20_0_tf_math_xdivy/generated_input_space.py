import tensorflow as tf
from dataclasses import dataclass

valid_test_case = {
    'inputs': [
        tf.constant([1.0, 0.0, 2.0]),
        tf.constant([2.0, 1.0, 0.0])
    ],
    'name': None
}

@dataclass
class InputSpace:
    name: list = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = [None, "xdivy_operation", "my_xdivy_op", "custom_div", "test_xdivy"]