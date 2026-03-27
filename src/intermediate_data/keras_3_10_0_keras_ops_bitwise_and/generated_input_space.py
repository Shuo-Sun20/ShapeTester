import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
x = keras.random.randint(shape=(3, 4), minval=0, maxval=10)
y = keras.random.randint(shape=(3, 4), minval=0, maxval=10)
valid_test_case = {
    "inputs": [x, y]
}

# Task 2, 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    inputs: List = field(default_factory=lambda: [
        # Shape combinations that should broadcast correctly
        # 1. Same shape (3x4)
        [
            keras.random.randint(shape=(3, 4), minval=0, maxval=10),
            keras.random.randint(shape=(3, 4), minval=0, maxval=10)
        ],
        # 2. Scalar broadcasting
        [
            keras.random.randint(shape=(), minval=0, maxval=10),
            keras.random.randint(shape=(3, 4), minval=0, maxval=10)
        ],
        # 3. Both scalars
        [
            keras.random.randint(shape=(), minval=0, maxval=10),
            keras.random.randint(shape=(), minval=0, maxval=10)
        ],
        # 4. Row vector broadcasting
        [
            keras.random.randint(shape=(1, 4), minval=0, maxval=10),
            keras.random.randint(shape=(3, 4), minval=0, maxval=10)
        ],
        # 5. Column vector broadcasting
        [
            keras.random.randint(shape=(3, 1), minval=0, maxval=10),
            keras.random.randint(shape=(3, 4), minval=0, maxval=10)
        ]
    ])