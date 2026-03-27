import keras
from dataclasses import dataclass, field

def call_func(inputs):
    return keras.ops.sin(inputs)

valid_test_case = {
    "inputs": keras.random.normal(shape=(3, 4))
}

@dataclass
class InputSpace:
    pass