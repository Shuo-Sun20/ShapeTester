import numpy as np
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "factor": 0.2,
    "fill_mode": "reflect",
    "interpolation": "bilinear",
    "seed": 42,
    "fill_value": 0.0,
    "data_format": "channels_last",
    "inputs": np.random.randn(2, 32, 32, 3).astype(np.float32),
    "training": True
}

# Task 2 & 3: Identify parameters affecting output shape and their value spaces
# According to the documentation and keras behavior, only 'data_format' affects the
# interpretation of input shape dimensions, which indirectly affects output shape.

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    data_format: List[str] = field(default_factory=lambda: ["channels_last", "channels_first"])