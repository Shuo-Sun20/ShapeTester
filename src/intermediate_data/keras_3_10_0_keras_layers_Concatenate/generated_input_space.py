import keras
import numpy as np
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "axis": 1,
    "inputs": [np.random.randn(2, 2, 5), np.random.randn(2, 1, 5)]
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [-3, -1, 0, 1, 2])