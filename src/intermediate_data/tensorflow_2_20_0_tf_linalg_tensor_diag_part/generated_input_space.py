import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, name=None):
    return tf.linalg.tensor_diag_part(input=inputs, name=name)

valid_test_case = {
    'inputs': tf.constant([[[[1111, 1112], [1121, 1122]],
                            [[1211, 1212], [1221, 1222]]],
                           [[[2111, 2112], [2121, 2122]],
                            [[2211, 2212], [2221, 2222]]]]),
    'name': None
}

@dataclass
class InputSpace:
    name: List[Optional[str]] = field(default_factory=lambda: [None, 'test', 'diag_op', 'tensor_diag', 'operation'])