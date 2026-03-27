import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define a valid test case dictionary
valid_test_case = {
    "inputs": [
        tf.random.uniform(shape=(2, 3, 4)),
        tf.random.uniform(shape=(2, 3))
    ],
    "name": None,
    "k": 0,
    "align": "RIGHT_LEFT"
}

# 2 & 3. Parameters affecting output shape: k and align
# k can be integer or tuple, discretized to 5 values including boundaries
k_values = [-2, -1, 0, 1, 2]  # Main diagonal and immediate neighbors

# align has 4 possible discrete values
align_values = ["RIGHT_LEFT", "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    k: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [-2, -1, 0, 1, 2]
    )
    align: List[str] = field(
        default_factory=lambda: ["RIGHT_LEFT", "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]
    )