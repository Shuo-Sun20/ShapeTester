import keras
from dataclasses import dataclass, field
from typing import List

# 1. Valid test case definition
valid_test_case = {
    "n": 3,
    "inputs": keras.random.normal(shape=(2, 32))
}

# 2 & 3. Only 'n' affects output shape
# Discretized value space for n (max 5 values including boundary)
# Boundary: 0 (minimum valid), 1, typical: 2, 3, 5

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    n: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 5])