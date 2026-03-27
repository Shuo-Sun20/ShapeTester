import numpy as np
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": np.random.rand(2, 3, 4),
    "mask": None,
    "data_format": "channels_last",
    "keepdims": False
}

@dataclass
class InputSpace:
    data_format: list = field(default_factory=lambda: ["channels_last", "channels_first"])
    keepdims: list = field(default_factory=lambda: [True, False])