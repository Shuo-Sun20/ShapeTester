import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, Union, List

valid_test_case = {
    "inputs": tf.constant([[0, 1, 0], [1, 1, 0]]),
    "axis": 0,
    "keepdims": False,
    "dtype": tf.int64,
    "name": None
}

@dataclass
class InputSpace:
    axis: List[Optional[Union[int, List[int]]]] = field(
        default_factory=lambda: [None, 0, 1, [0, 1], [-1]]
    )
    keepdims: List[bool] = field(
        default_factory=lambda: [False, True]
    )