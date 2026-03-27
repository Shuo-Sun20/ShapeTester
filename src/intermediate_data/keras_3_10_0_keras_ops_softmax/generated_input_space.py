import numpy as np
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": np.random.randn(3, 4),
    "axis": 1
}

@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [-3, -2, -1, 0, 1, 2])