import tensorflow as tf
from dataclasses import dataclass, field

def call_func(inputs, axis=-1, name=None):
    return tf.math.softmax(logits=inputs, axis=axis, name=name)

example_input = tf.random.normal(shape=(3, 4))
example_output = call_func(inputs=example_input)

valid_test_case = {
    "inputs": example_input,
    "axis": -1,
    "name": None
}

@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [-2, -1, 0, 1, [0, 1]])