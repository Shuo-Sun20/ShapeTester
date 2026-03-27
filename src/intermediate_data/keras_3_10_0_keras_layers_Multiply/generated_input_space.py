from dataclasses import dataclass, field
import numpy as np

# Task 1: Define valid_test_case
x1 = np.random.rand(2, 3, 4).astype('float32')
x2 = np.random.rand(2, 3, 4).astype('float32')
valid_test_case = {
    "inputs": [x1, x2],
    "name": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    name: list = field(default_factory=lambda: [
        None,
        "multiply_layer",
        "custom_name",
        "layer_1",
        ""
    ])