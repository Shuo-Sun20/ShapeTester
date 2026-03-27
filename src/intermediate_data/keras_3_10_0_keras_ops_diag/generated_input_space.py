import keras
from dataclasses import dataclass, field
from typing import List

# Valid test case as required
valid_test_case = {
    "inputs": [keras.random.normal(shape=(4, 4))],
    "k": 0
}

@dataclass
class InputSpace:
    k: List[int] = field(default_factory=lambda: [-4, -2, -1, 0, 1, 2, 4])