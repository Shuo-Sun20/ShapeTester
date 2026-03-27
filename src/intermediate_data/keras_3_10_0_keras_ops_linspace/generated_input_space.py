import keras
from dataclasses import dataclass, field
from typing import List, Optional

# Valid test case dictionary
valid_test_case = {
    "inputs": [keras.ops.array(0.0), keras.ops.array(10.0)],
    "num": 10,
    "endpoint": True,
    "retstep": False,
    "dtype": None,
    "axis": 0
}

# Parameters affecting output shape: num, axis, and the shapes of inputs
@dataclass
class InputSpace:
    num: List[int] = field(default_factory=lambda: [0, 1, 2, 10, 50, 100, 200])
    axis: List[int] = field(default_factory=lambda: [-2, -1, 0, 1, 2])