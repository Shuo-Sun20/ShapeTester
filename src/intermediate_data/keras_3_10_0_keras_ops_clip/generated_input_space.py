import keras
import numpy as np
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": [keras.random.uniform((3, 4), -2, 2)],
    "x_min": -1.0,
    "x_max": 1.0
}

@dataclass
class InputSpace:
    x_min: list = field(default_factory=lambda: [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0])
    x_max: list = field(default_factory=lambda: [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0])