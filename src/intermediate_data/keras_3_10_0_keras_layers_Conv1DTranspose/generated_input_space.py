import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Union

valid_test_case = {
    "filters": 32,
    "kernel_size": 3,
    "inputs": np.random.rand(4, 10, 128),
    "strides": 2,
    "padding": "valid",
    "output_padding": None,
    "data_format": None,
    "dilation_rate": 1,
    "activation": 'relu',
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None
}

@dataclass
class InputSpace:
    filters: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    kernel_size: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    strides: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    output_padding: List[Optional[Union[int, List[int]]]] = field(default_factory=lambda: [None, 0, 1, 2, 3])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    dilation_rate: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5])