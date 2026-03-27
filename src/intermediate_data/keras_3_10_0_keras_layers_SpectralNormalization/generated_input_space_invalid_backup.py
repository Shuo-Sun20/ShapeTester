import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "layer": keras.layers.Conv2D(filters=2, kernel_size=2),
    "inputs": np.random.randn(1, 10, 10, 1).astype('float32'),
    "power_iterations": 1
}

@dataclass
class InputSpace:
    layer: List[keras.layers.Layer] = field(
        default_factory=lambda: [
            keras.layers.Dense(units=4),
            keras.layers.Dense(units=8),
            keras.layers.Dense(units=16),
            keras.layers.Dense(units=32),
            keras.layers.Dense(units=64)
        ]
    )