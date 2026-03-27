import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "height": 224,
    "width": 224,
    "seed": 42,
    "data_format": None,
    "name": None,
    "dtype": None,
    "inputs": keras.random.normal(shape=(4, 256, 256, 3))
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    height: List[int] = field(default_factory=lambda: [32, 64, 128, 224, 256])
    width: List[int] = field(default_factory=lambda: [32, 64, 128, 224, 256])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, 'channels_last', 'channels_first'])