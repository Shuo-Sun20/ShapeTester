import keras
import numpy as np
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
x = keras.ops.convert_to_tensor(np.random.randint(0, 100, (3, 4)))
y = keras.ops.convert_to_tensor(np.random.randint(0, 100, (3, 4)))
valid_test_case = {"inputs": [x, y]}

# Tasks 2-4: Define InputSpace
@dataclass
class InputSpace:
    # The only parameter is 'inputs', which affects output shape
    inputs: list = field(default_factory=lambda: [
        [keras.ops.convert_to_tensor(np.random.randint(0, 100, ())),
         keras.ops.convert_to_tensor(np.random.randint(0, 100, ()))],
        [keras.ops.convert_to_tensor(np.random.randint(0, 100, (5,))),
         keras.ops.convert_to_tensor(np.random.randint(0, 100, (5,)))],
        [keras.ops.convert_to_tensor(np.random.randint(0, 100, (3, 4))),
         keras.ops.convert_to_tensor(np.random.randint(0, 100, (3, 4)))],
        [keras.ops.convert_to_tensor(np.random.randint(0, 100, (2, 3, 4))),
         keras.ops.convert_to_tensor(np.random.randint(0, 100, (2, 3, 4)))],
        [keras.ops.convert_to_tensor(np.random.randint(0, 100, (1, 4))),
         keras.ops.convert_to_tensor(np.random.randint(0, 100, (3, 1)))]
    ])