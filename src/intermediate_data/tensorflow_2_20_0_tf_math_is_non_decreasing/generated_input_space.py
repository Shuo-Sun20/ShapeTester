import tensorflow as tf
from dataclasses import dataclass

valid_test_case = {
    'inputs': tf.random.uniform(shape=(5,), minval=0, maxval=10, dtype=tf.float32),
    'name': 'is_non_decreasing'
}

@dataclass
class InputSpace:
    name: list = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = [None, '', 'test_name', 'is_non_decreasing', 'custom_name']