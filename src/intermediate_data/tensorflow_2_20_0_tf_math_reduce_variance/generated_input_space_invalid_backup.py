import tensorflow as tf
from dataclasses import dataclass

valid_test_case = {
    'inputs': tf.random.normal(shape=(2, 3)),
    'axis': 1,
    'keepdims': True,
    'name': None
}

@dataclass
class InputSpace:
    axis: list = None
    keepdims: list = None
    
    def __post_init__(self):
        if self.axis is None:
            self.axis = [None, 0, -1, [0, 1], [1, 2]]
        if self.keepdims is None:
            self.keepdims = [True, False]