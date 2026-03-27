import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": tf.constant(np.random.rand(2, 3, 4), dtype=tf.float32),
    "k": 1,
    "padding_value": 0,
    "align": "RIGHT_LEFT",
    "name": None
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    k: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            0,                    # main diagonal
            1,                    # first superdiagonal
            -1,                   # first subdiagonal
            (-1, 1),              # band around main diagonal
            (-2, 2)               # wider band
        ]
    )
    align: List[str] = field(
        default_factory=lambda: [
            "RIGHT_LEFT",         # default alignment
            "LEFT_RIGHT",
            "LEFT_LEFT",
            "RIGHT_RIGHT"
        ]
    )