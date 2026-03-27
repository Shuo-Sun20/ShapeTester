from dataclasses import dataclass
import numpy as np

valid_test_case = {
    "inputs": np.random.rand(2, 64, 64, 3) * 255.0,
    "factor": 0.5,
    "value_range": (0, 255),
    "data_format": None,
    "seed": None
}

@dataclass
class InputSpace:
    data_format: list = None
    
    def __post_init__(self):
        if self.data_format is None:
            self.data_format = [None, 'channels_last', 'channels_first']