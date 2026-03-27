import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

def call_func(inputs, factor=1.0, kernel_size=3, sigma=1.0, value_range=(0, 255), data_format=None, seed=None):
    layer = keras.layers.RandomGaussianBlur(
        factor=factor,
        kernel_size=kernel_size,
        sigma=sigma,
        value_range=value_range,
        data_format=data_format,
        seed=seed
    )
    return layer(inputs)

valid_test_case = {
    'inputs': np.random.uniform(0, 255, size=(4, 32, 32, 3)).astype('float32'),
    'factor': 1.0,
    'kernel_size': 3,
    'sigma': 1.0,
    'value_range': (0, 255),
    'data_format': None,
    'seed': None
}

@dataclass
class InputSpace:
    data_format: list[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])