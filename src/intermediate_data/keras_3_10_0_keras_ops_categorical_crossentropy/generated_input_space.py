import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, from_logits=False, axis=-1):
    target, output = inputs
    return keras.ops.categorical_crossentropy(target, output, from_logits=from_logits, axis=axis)

np.random.seed(0)
target_tensor = keras.ops.convert_to_tensor(np.eye(3))
output_tensor = keras.ops.convert_to_tensor(np.random.dirichlet([1, 1, 1], size=3))

valid_test_case = {
    "inputs": [target_tensor, output_tensor],
    "from_logits": False,
    "axis": -1
}

@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [-2, -1, 0, 1, 2])