import tensorflow as tf
from dataclasses import dataclass
from typing import List, Union, Optional

def call_func(inputs, name=None):
    return tf.nn.selu(features=inputs, name=name)

example_input = tf.random.normal(shape=(3, 4))
valid_test_case = {"inputs": example_input, "name": None}

@dataclass
class InputSpace:
    name: List[Optional[str]] = None

    def __post_init__(self):
        if self.name is None:
            self.name = [None, "selu_op1", "selu_op2", "selu_op3", "selu_op4"]