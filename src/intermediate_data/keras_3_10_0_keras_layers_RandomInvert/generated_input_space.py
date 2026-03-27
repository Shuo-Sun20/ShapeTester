import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

valid_test_case = {
    "factor": (0.3, 0.7),
    "value_range": (0, 255),
    "seed": 42,
    "data_format": None,
    "inputs": np.random.uniform(0, 255, size=(4, 32, 32, 3)).astype(np.float32)
}

@dataclass
class InputSpace:
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])