import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Optional

# Define the call_func as per the original code
def call_func(inputs, alpha=20.0, sigma=5.0, interpolation="bilinear", fill_mode="reflect", fill_value=0.0, seed=None, data_format=None):
    images = inputs[0]
    return keras.ops.image.elastic_transform(images, alpha, sigma, interpolation, fill_mode, fill_value, seed, data_format)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [np.random.random((64, 80, 3)).astype("float32")],
    "alpha": 20.0,
    "sigma": 5.0,
    "interpolation": "bilinear",
    "fill_mode": "reflect",
    "fill_value": 0.0,
    "seed": None,
    "data_format": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Defines parameters that can affect the output shape.
    Since keras.ops.image.elastic_transform maintains the input shape for all parameters,
    this class only includes the data_format parameter which influences interpretation
    of the input tensor shape.
    """
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [
            None,                 # Default (uses keras.config.image_data_format)
            "channels_last",      # Shape: (batch, height, width, channels)
            "channels_first",     # Shape: (batch, channels, height, width)
        ]
    )