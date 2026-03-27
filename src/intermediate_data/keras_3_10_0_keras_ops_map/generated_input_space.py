import keras
from dataclasses import dataclass, field
from typing import Callable, List

# Define function variations that affect output shape
def identity(x):
    return x

def scalar_output(x):
    return keras.ops.sum(x)

def flatten_output(x):
    return keras.ops.flatten(x)

def reshaped_output(x):
    return keras.ops.reshape(x, (1, 9))

def elementwise_double(x):
    return x * 2

def nested_output(x):
    return {"y1": x**2, "y2": x * 10}

# Define the valid test case
valid_test_case = {
    "f": elementwise_double,
    "inputs": keras.random.normal(shape=(5, 3, 3))
}

# Define InputSpace dataclass
@dataclass
class InputSpace:
    f: List[Callable] = field(default_factory=lambda: [
        identity,
        scalar_output,
        flatten_output,
        reshaped_output,
        elementwise_double,
        nested_output
    ])