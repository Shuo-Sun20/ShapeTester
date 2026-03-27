import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, List

def call_func(inputs, approximate=False, name=None):
    return tf.nn.gelu(features=inputs, approximate=approximate, name=name)

# 1. Define valid_test_case
example_input = tf.random.normal(shape=(2, 3))
valid_test_case = {
    'inputs': example_input,
    'approximate': True,
    'name': 'gelu_operation'
}

# 2. Identify shape-affecting parameters (none except inputs)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    approximate: List[Union[bool, None]] = field(default_factory=lambda: [True, False])
    name: List[Union[str, None]] = field(default_factory=lambda: [None, 'gelu_operation', 'gelu', 'activation', 'custom_gelu'])