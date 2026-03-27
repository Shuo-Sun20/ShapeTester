import numpy as np
from dataclasses import dataclass, field
from typing import Optional

valid_test_case = {
    "factor": (0.2, 0.8),
    "inputs": np.random.rand(2, 224, 224, 3).astype(np.float32),
    "value_range": (0, 255),
    "data_format": None,
    "seed": 42
}

@dataclass
class InputSpace:
    data_format: list = field(default_factory=lambda: [None, "channels_last", "channels_first"])