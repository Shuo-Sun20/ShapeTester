import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, value_range=(0, 255)):
    layer = keras.layers.AutoContrast(value_range=value_range)
    return layer(inputs)

valid_test_case = {
    'inputs': np.random.uniform(0, 255, size=(2, 32, 32, 3)).astype(np.float32),
    'value_range': (0, 255)
}

@dataclass
class InputSpace:
    value_range: list = field(
        default_factory=lambda: [
            (0, 1),
            (0, 255),
            (0, 127),
            (-1, 1),
            (0, 65535)
        ]
    )