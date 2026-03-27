import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case dictionary
valid_test_case = {
    "height_factor": 0.2,
    "width_factor": (-0.1, 0.1),
    "inputs": np.random.rand(4, 32, 32, 3).astype(np.float32),
    "fill_mode": "constant",
    "interpolation": "bilinear",
    "seed": 42,
    "fill_value": 0.5,
    "data_format": "channels_last"
}

# 2. Parameters affecting output shape: height_factor, width_factor
# Note: data_format affects dimension ordering but not spatial dimensions

# 4. Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    # height_factor: float or tuple of two floats
    height_factor: List[Union[float, Tuple[float, float]]] = field(
        default_factory=lambda: [
            -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5,  # single values
            (-0.5, -0.3), (-0.3, -0.1), (-0.2, 0.0), (-0.1, 0.1), (0.0, 0.2),  # negative ranges
            (-0.2, 0.2), (-0.1, 0.3), (0.1, 0.5),  # mixed ranges
            (0.2, 0.4), (0.3, 0.5)  # positive ranges
        ]
    )
    
    # width_factor: float or tuple of two floats
    width_factor: List[Union[float, Tuple[float, float]]] = field(
        default_factory=lambda: [
            -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5,  # single values
            (-0.5, -0.3), (-0.3, -0.1), (-0.2, 0.0), (-0.1, 0.1), (0.0, 0.2),  # negative ranges
            (-0.2, 0.2), (-0.1, 0.3), (0.1, 0.5),  # mixed ranges
            (0.2, 0.4), (0.3, 0.5)  # positive ranges
        ]
    )