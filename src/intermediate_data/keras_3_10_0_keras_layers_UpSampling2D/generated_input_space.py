import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

def call_func(size=(2, 2), data_format=None, interpolation="nearest", inputs=None):
    layer = keras.layers.UpSampling2D(
        size=size,
        data_format=data_format,
        interpolation=interpolation
    )
    output = layer(inputs)
    return output

# 1. Valid test case for call_func()
valid_test_case = {
    "size": (2, 2),
    "data_format": "channels_last",
    "interpolation": "bilinear",
    "inputs": np.random.randn(2, 4, 4, 3).astype(np.float32)
}

# 2. Parameters affecting output shape: size, data_format
# 3. Discretized value spaces

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape of UpSampling2D."""
    
    size: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 1),     # No upsampling
            (2, 2),     # Default value
            (3, 3),     # Larger symmetric factor
            (2, 3),     # Asymmetric factors
            (4, 1)      # Single-dimension upsampling
        ]
    )
    
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [
            "channels_last",    # Default
            "channels_first",   # Alternative
            None                # Uses keras.config default
        ]
    )