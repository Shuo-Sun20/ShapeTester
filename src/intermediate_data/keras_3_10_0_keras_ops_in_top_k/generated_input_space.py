import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List

def call_func(inputs, k):
    targets, predictions = inputs
    return keras.ops.in_top_k(targets, predictions, k)

valid_test_case = {
    "inputs": [
        keras.ops.convert_to_tensor(np.array([2, 5, 3], dtype=np.int32)),
        keras.ops.convert_to_tensor(
            np.array([[0.1, 0.4, 0.6, 0.9, 0.5],
                      [0.1, 0.7, 0.9, 0.8, 0.3],
                      [0.1, 0.6, 0.9, 0.9, 0.5]], dtype=np.float32)
        )
    ],
    "k": 3
}

@dataclass
class InputSpace:
    k: List[Union[int, np.int32, np.int64]] = field(default_factory=lambda: [1, 2, 3, 4, 5, 10, 0, -1])