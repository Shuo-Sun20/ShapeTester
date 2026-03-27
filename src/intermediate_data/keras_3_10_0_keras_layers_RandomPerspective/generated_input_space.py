import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    "inputs": np.random.rand(2, 32, 32, 3).astype(np.float32),
    "factor": 0.5,
    "scale": 0.3,
    "interpolation": "bilinear",
    "fill_value": 0.0,
    "seed": None,
    "data_format": None
}

# Task 2: Parameters that can affect output shape: data_format
# Task 3 & 4: Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])