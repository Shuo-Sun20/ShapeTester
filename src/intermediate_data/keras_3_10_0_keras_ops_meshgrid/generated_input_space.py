import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, indexing="xy"):
    return keras.ops.meshgrid(*inputs, indexing=indexing)

# Generate random 1D tensors
x = keras.ops.convert_to_tensor(np.random.randn(3))
y = keras.ops.convert_to_tensor(np.random.randn(4))
z = keras.ops.convert_to_tensor(np.random.randn(5))

example_output = call_func(inputs=[x, y, z], indexing="ij")

valid_test_case = {
    "inputs": [x, y, z],
    "indexing": "ij"
}

@dataclass
class InputSpace:
    indexing: list[str] = field(default_factory=lambda: ["xy", "ij"])