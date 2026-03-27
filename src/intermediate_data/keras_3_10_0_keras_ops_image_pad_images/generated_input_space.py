import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    "inputs": [np.random.random((2, 15, 25, 3))],
    "top_padding": 2,
    "left_padding": 3,
    "bottom_padding": None,
    "right_padding": None,
    "target_height": 20,
    "target_width": 30,
    "data_format": None
}

@dataclass
class InputSpace:
    top_padding: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 5, 10])
    left_padding: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 3, 5, 10])
    bottom_padding: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 5, 10])
    right_padding: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, 2, 5, 10])
    target_height: List[Optional[int]] = field(default_factory=lambda: [None, 15, 16, 20, 25, 30])
    target_width: List[Optional[int]] = field(default_factory=lambda: [None, 25, 26, 30, 35, 40])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])