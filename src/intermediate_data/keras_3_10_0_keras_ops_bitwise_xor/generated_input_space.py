import keras
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "inputs": [
        keras.random.randint(shape=(2, 3), minval=0, maxval=10),
        keras.random.randint(shape=(2, 3), minval=0, maxval=10)
    ]
}

@dataclass
class InputSpace:
    inputs: List[List] = field(default_factory=lambda: [
        [
            keras.random.randint(shape=(2, 3), minval=0, maxval=10),
            keras.random.randint(shape=(2, 3), minval=0, maxval=10)
        ],
        [
            keras.random.randint(shape=(), minval=0, maxval=10),
            keras.random.randint(shape=(), minval=0, maxval=10)
        ],
        [
            keras.random.randint(shape=(5,), minval=0, maxval=10),
            keras.random.randint(shape=(5,), minval=0, maxval=10)
        ],
        [
            keras.random.randint(shape=(1, 4), minval=0, maxval=10),
            keras.random.randint(shape=(3, 1), minval=0, maxval=10)
        ],
        [
            keras.random.randint(shape=(2, 3, 4), minval=0, maxval=10),
            keras.random.randint(shape=(2, 3, 4), minval=0, maxval=10)
        ]
    ])