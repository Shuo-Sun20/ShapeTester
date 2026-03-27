import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, List

def call_func(inputs, height_factor, width_factor=None, fill_mode="reflect", interpolation="bilinear", seed=None, fill_value=0.0, data_format=None):
    layer = keras.layers.RandomZoom(
        height_factor=height_factor,
        width_factor=width_factor,
        fill_mode=fill_mode,
        interpolation=interpolation,
        seed=seed,
        fill_value=fill_value,
        data_format=data_format
    )
    return layer(inputs)

# Valid test case
valid_test_case = {
    "inputs": np.random.rand(5, 224, 224, 3),
    "height_factor": (0.2, 0.3),
    "width_factor": (-0.2, 0.2),
    "fill_mode": "reflect",
    "interpolation": "bilinear",
    "seed": 42,
    "fill_value": 0.0,
    "data_format": "channels_last"
}

@dataclass
class InputSpace:
    data_format: List[Optional[str]] = field(default_factory=lambda: ["channels_last", "channels_first"])