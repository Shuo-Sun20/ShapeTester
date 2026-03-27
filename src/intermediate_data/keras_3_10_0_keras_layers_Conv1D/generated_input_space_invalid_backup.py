import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Optional

# 1. Define valid_test_case
valid_test_case = {
    "filters": 32,
    "kernel_size": 3,
    "strides": 1,
    "padding": "valid",
    "data_format": None,
    "dilation_rate": 1,
    "groups": 1,
    "activation": 'relu',
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "inputs": np.random.rand(4, 10, 8).astype(np.float32)
}

# 2,3,4. Define InputSpace dataclass with parameters that affect output shape
@dataclass
class InputSpace:
    filters: List[int] = field(default_factory=lambda: [1, 4, 16, 64, 256])
    kernel_size: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    strides: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: List[str] = field(default_factory=lambda: ["valid", "same", "causal"])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    dilation_rate: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])