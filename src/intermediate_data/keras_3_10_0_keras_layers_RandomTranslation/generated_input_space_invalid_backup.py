import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple

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

# 2. Parameters affecting output shape: height_factor, width_factor, data_format

@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output tensor shape,
    with discretized value ranges (max 5 values per parameter)."""
    
    height_factor: List[Union[float, Tuple[float, float]]] = field(
        default_factory=lambda: [
            -0.5,          # Negative bound
            -0.25,         # Negative typical
            0.0,           # Zero/no shift
            0.25,          # Positive typical
            0.5,           # Positive bound
        ]
    )
    
    width_factor: List[Union[float, Tuple[float, float]]] = field(
        default_factory=lambda: [
            -0.5,          # Negative bound
            -0.25,         # Negative typical
            0.0,           # Zero/no shift
            0.25,          # Positive typical
            0.5,           # Positive bound
        ]
    )
    
    data_format: List[str] = field(
        default_factory=lambda: [
            "channels_last",
            "channels_first",
        ]
    )