import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.rand(4, 10, 10, 128).astype(np.float32),
    "filters": 32,
    "kernel_size": 3,
    "strides": (1, 1),
    "padding": "valid",
    "data_format": None,
    "dilation_rate": (1, 1),
    "groups": 1,
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

# Task 2 & 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    filters: List[int] = field(default_factory=lambda: [1, 16, 32, 64, 128])
    kernel_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 3, 5, 7, (3,5)]
    )
    strides: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, (1,2), (2,1), (2,2)]
    )
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )
    dilation_rate: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, (1,2), (2,1), (2,2)]
    )
    groups: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])