import numpy as np
import keras
from dataclasses import dataclass, field

# 1. Define valid_test_case
valid_test_case = {
    "inputs": np.random.random((2, 64, 80, 3)),
    "transform": np.array(
        [
            [1.5, 0, -20, 0, 1.5, -16, 0, 0],
            [1, 0, -20, 0, 1, -16, 0, 0],
        ]
    ),
    "interpolation": "bilinear",
    "fill_mode": "constant",
    "fill_value": 0,
    "data_format": None,
}

# 2 & 3. Identify shape-affecting parameters and their value spaces
# data_format is the only parameter affecting output shape (through dimension reordering)
# It is discrete with possible values: None, "channels_last", "channels_first"

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    data_format: list = field(default_factory=lambda: [None, "channels_last", "channels_first"])