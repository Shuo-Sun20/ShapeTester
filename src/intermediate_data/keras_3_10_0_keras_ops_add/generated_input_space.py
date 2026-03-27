import keras
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [
        keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 3))),
        keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 3)))
    ]
}

# Tasks 2, 3, and 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # The only parameter in call_func's signature is 'inputs'
    # We need to generate different tensor pairs for testing
    inputs: list = field(default_factory=lambda: [
        # Case 1: Same shape (2D)
        [
            keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 3))),
            keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 3)))
        ],
        # Case 2: Scalar addition
        [
            keras.ops.convert_to_tensor(keras.random.normal(shape=(1,))),
            keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 3)))
        ],
        # Case 3: Broadcast row vector
        [
            keras.ops.convert_to_tensor(keras.random.normal(shape=(1, 3))),
            keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 1)))
        ],
        # Case 4: Broadcast column vector
        [
            keras.ops.convert_to_tensor(keras.random.normal(shape=(2, 1))),
            keras.ops.convert_to_tensor(keras.random.normal(shape=(1, 3)))
        ],
        # Case 5: 3D broadcasting
        [
            keras.ops.convert_to_tensor(keras.random.normal(shape=(4, 1, 2))),
            keras.ops.convert_to_tensor(keras.random.normal(shape=(1, 3, 2)))
        ]
    ])