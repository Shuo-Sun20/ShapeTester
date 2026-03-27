import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Valid test case
valid_test_case = {
    "pool_size": 3,
    "inputs": np.random.rand(2, 10, 3).astype('float32'),
    "strides": 2,
    "padding": "same",
    "data_format": "channels_last",
    "name": None
}

# 2 & 3. Parameters affecting output shape with discretized value spaces
@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting AveragePooling1D output shape."""
    pool_size: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 7])
    strides: List[Optional[int]] = field(default_factory=lambda: [None, 1, 2, 3, 4])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[str] = field(default_factory=lambda: ["channels_last", "channels_first"])