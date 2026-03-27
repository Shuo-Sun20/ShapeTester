import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Valid test case
valid_test_case = {
    "height": 32,
    "width": 32,
    "inputs": np.random.rand(2, 64, 64, 3).astype(np.float32),
    "interpolation": "bilinear",
    "crop_to_aspect_ratio": False,
    "pad_to_aspect_ratio": False,
    "fill_mode": "constant",
    "fill_value": 0.0,
    "antialias": False,
    "data_format": None,
    "name": None,
    "dtype": None,
    "trainable": None
}

# 2 & 3. Parameters affecting output shape and their discretized value spaces
@dataclass
class InputSpace:
    height: List[int] = field(default_factory=lambda: [1, 32, 64, 128, 256])
    width: List[int] = field(default_factory=lambda: [1, 32, 64, 128, 256])
    crop_to_aspect_ratio: List[bool] = field(default_factory=lambda: [True, False])
    pad_to_aspect_ratio: List[bool] = field(default_factory=lambda: [True, False])
    data_format: List[str] = field(default_factory=lambda: ["channels_last", "channels_first"])