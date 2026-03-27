import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

# 1. Define a valid test case dictionary
valid_test_case = {
    "inputs": [tf.random.uniform(shape=(3, 4), minval=1, maxval=5, dtype=tf.float32)],
    "axis": 1,
    "exclusive": True,
    "reverse": False,
    "name": None
}

# 2. & 3. Parameters that affect output shape: axis (only parameter affecting shape besides inputs)
# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    axis: List[int] = field(default_factory=lambda: [-2, -1, 0, 1])