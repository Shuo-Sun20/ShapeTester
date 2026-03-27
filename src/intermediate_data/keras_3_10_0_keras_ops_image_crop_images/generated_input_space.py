import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": np.random.randn(4, 32, 32, 3).astype("float32"),
    "top_cropping": 2,
    "left_cropping": 3,
    "bottom_cropping": 4,
    "right_cropping": 5,
    "target_height": None,
    "target_width": None,
    "data_format": "channels_last"
}

# 2. Parameters affecting output shape: top_cropping, left_cropping, bottom_cropping, right_cropping, target_height, target_width

@dataclass
class InputSpace:
    top_cropping: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    left_cropping: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bottom_cropping: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    right_cropping: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    target_height: List[Optional[int]] = field(default_factory=lambda: [None, 1, 8, 16, 24, 32, 48, 64])
    target_width: List[Optional[int]] = field(default_factory=lambda: [None, 1, 8, 16, 24, 32, 48, 64])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])