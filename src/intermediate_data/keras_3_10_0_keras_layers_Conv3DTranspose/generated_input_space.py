import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

# 1. Define valid_test_case variable
valid_test_case = {
    "filters": 32,
    "kernel_size": 2,
    "strides": 2,
    "padding": "valid",
    "data_format": None,
    "output_padding": None,
    "dilation_rate": 1,
    "activation": "relu",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "inputs": np.random.rand(4, 10, 8, 12, 128).astype('float32')
}

# 2 & 3 & 4: InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output shape with discretized value spaces"""
    
    filters: List[int] = field(default_factory=lambda: [1, 2, 8, 16, 32, 64, 128, 256])
    kernel_size: List[Union[int, tuple]] = field(default_factory=lambda: [
        1, 2, 3, 5, 7, 
        (1, 1, 1), (2, 2, 2), (3, 3, 3), 
        (1, 2, 3), (3, 2, 1)
    ])
    strides: List[Union[int, tuple]] = field(default_factory=lambda: [
        1, 2, 3, 4, 
        (1, 1, 1), (2, 2, 2), (3, 3, 3),
        (1, 2, 1), (2, 1, 2)
    ])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    output_padding: List[Optional[Union[int, tuple]]] = field(default_factory=lambda: [
        None, 0, 1, 2,
        (0, 0, 0), (1, 1, 1), (2, 2, 2),
        (0, 1, 0), (1, 0, 1)
    ])
    dilation_rate: List[Union[int, tuple]] = field(default_factory=lambda: [
        1, 2, 3, 
        (1, 1, 1), (2, 2, 2), (3, 3, 3),
        (1, 2, 1), (2, 1, 2)
    ])

# Example instantiation
var = InputSpace()