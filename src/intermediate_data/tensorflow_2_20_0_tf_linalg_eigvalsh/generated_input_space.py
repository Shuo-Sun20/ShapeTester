import tensorflow as tf
from dataclasses import dataclass
from typing import List, Optional

def call_func(inputs, name=None):
    tensor = inputs[0]
    result = tf.linalg.eigvalsh(tensor=tensor, name=name)
    return result

rand_tensor = tf.random.normal(shape=(3, 3))
symmetric_tensor = 0.5 * (rand_tensor + tf.transpose(rand_tensor))
valid_test_case = {'inputs': [symmetric_tensor], 'name': None}

@dataclass
class InputSpace:
    name: List[Optional[str]] = None  # name parameter has no effect on output shape
    
    def __post_init__(self):
        if self.name is None:
            self.name = ['test_op_1', 'test_op_2', None, '', 'eigvalsh_op']  # discretized name values