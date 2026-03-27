import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

valid_test_case = {
    "inputs": np.random.random((2, 64, 80, 3)).astype(np.float32),
    "kernel_size": (3, 3),
    "sigma": (1.0, 1.0),
    "data_format": None
}

@dataclass
class InputSpace:
    kernel_size: List[Tuple[int, int]] = field(default_factory=lambda: [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9), (11, 11), (3, 5), (5, 3)])
    sigma: List[Tuple[float, float]] = field(default_factory=lambda: [(0.1, 0.1), (0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (5.0, 5.0), (0.5, 2.0), (2.0, 0.5)])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])