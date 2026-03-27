import keras
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

# 1. Define valid_test_case dictionary
random_input = np.random.rand(2, 32, 32, 3).astype(np.float32)
valid_test_case = {
    "inputs": random_input,
    "x_factor": (0.0, 0.1),
    "y_factor": 0.05,
    "interpolation": "bilinear",
    "fill_mode": "constant",
    "fill_value": 0.0,
    "data_format": None,
    "seed": 42
}

# 2. Parameters that affect output shape: data_format
# Note: Only data_format affects interpretation of shape dimensions
# No parameters actually change the tensor dimensions

# 3. Value space definitions

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """Parameters that could affect output shape interpretation."""
    
    # data_format: affects dimension interpretation but not actual shape
    # Included as it's the only parameter that relates to shape structure
    data_format: List[Optional[str]] = None
    
    def __post_init__(self):
        # Initialize with discretized values
        if self.data_format is None:
            self.data_format = [None, "channels_last", "channels_first"]
    
    # Note: x_factor, y_factor, interpolation, fill_mode, fill_value, seed
    # do NOT affect the output tensor shape - they only affect content