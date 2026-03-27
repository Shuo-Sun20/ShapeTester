import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "inputs": np.random.randn(4, 3).astype(np.float32),
    "threshold": 0.5,
    "default_value": 0.0
}

def call_func(inputs, threshold, default_value):
    return keras.ops.threshold(inputs, threshold, default_value)

@dataclass
class InputSpace:
    threshold: List[float] = field(default_factory=lambda: [-10.0, -5.0, -1.0, 0.0, 0.5, 1.0, 5.0, 10.0])
    default_value: List[float] = field(default_factory=lambda: [-10.0, -5.0, -1.0, 0.0, 0.5, 1.0, 5.0, 10.0])