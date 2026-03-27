import tensorflow as tf

def call_func(inputs, name=None):
    return tf.math.zero_fraction(value=inputs, name=name)

example_tensor = tf.constant([1.0, 0.0, 3.0, 0.0, 5.0])
valid_test_case = {'inputs': example_tensor, 'name': None}

from dataclasses import dataclass
from typing import List, Union

@dataclass
class InputSpace:
    name: List[Union[str, None]] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = [None, '', 'zero_fraction_op', 'sparsity_summary', 'test']