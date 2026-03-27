import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

valid_test_case = {
    "inputs": np.random.rand(4, 10, 12).astype(np.float32),
    "kernel_size": 3,
    "strides": 3,
    "padding": "valid",
    "depth_multiplier": 2,
    "data_format": None,
    "dilation_rate": 1,
    "activation": "relu",
    "use_bias": True,
    "depthwise_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "depthwise_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "depthwise_constraint": None,
    "bias_constraint": None
}

@dataclass
class InputSpace:
    kernel_size: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    strides: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    depth_multiplier: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    dilation_rate: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [1, 2, 4, 8, 16])