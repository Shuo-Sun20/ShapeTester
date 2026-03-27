import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "inputs": tf.random.normal(shape=(3, 4)),
    "beta": 1.0
}

@dataclass
class InputSpace:
    # There are no parameters other than "inputs" that affect the output shape.
    # Since "inputs" is excluded per instructions, InputSpace remains empty.
    pass