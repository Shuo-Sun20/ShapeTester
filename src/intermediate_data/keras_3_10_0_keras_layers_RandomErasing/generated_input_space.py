from dataclasses import dataclass, field
import numpy as np

valid_test_case = {
    "inputs": np.random.uniform(0, 255, size=(4, 32, 32, 3)).astype(np.float32),
    "factor": 0.5,
    "scale": (0.02, 0.33),
    "fill_value": 0.0,
    "value_range": (0, 255),
    "seed": 42,
    "data_format": "channels_last"
}

@dataclass
class InputSpace:
    data_format: list = field(default_factory=lambda: [None, "channels_first", "channels_last"])