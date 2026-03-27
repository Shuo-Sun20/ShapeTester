import keras
from dataclasses import dataclass, field
from typing import Union, List

# 1. Valid test case definition
valid_test_case = {
    "inputs": keras.random.normal(shape=(4, 4, 4)),
    "kth": 2,
    "axis": -1
}

# 4. InputSpace class definition
@dataclass
class InputSpace:
    kth: List[Union[int, List[int]]] = field(default_factory=lambda: [0, 1, 2, 3, [0, 2]])
    axis: List[Union[int, None]] = field(default_factory=lambda: [-3, -1, 0, 2, None])