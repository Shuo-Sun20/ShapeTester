import numpy as np
from dataclasses import dataclass

valid_test_case = {
    "inputs": np.random.uniform(0, 255, size=(2, 32, 32, 3)).astype(np.float32),
    "factor": 0.5,
    "scale": (0.8, 1.2),
    "interpolation": "bilinear",
    "fill_mode": "reflect",
    "fill_value": 0.0,
    "value_range": (0, 255),
    "seed": 42,
    "data_format": None
}

@dataclass
class InputSpace:
    data_format: list = None
    
    def __post_init__(self):
        if self.data_format is None:
            self.data_format = [None, "channels_last", "channels_first"]