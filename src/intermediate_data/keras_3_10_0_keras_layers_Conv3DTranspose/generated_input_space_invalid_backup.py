import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

# 1. Define valid_test_case variable
example_input = np.random.rand(4, 10, 8, 12, 128).astype('float32')
valid_test_case = {
    "filters": 32,
    "kernel_size": 2,
    "strides": 2,
    "padding": "valid",
    "data_format": None,
    "output_padding": None,
    "dilation_rate": (1, 1, 1),
    "activation": "relu",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "inputs": example_input
}

# 2. & 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # filters affects output channel dimension
    filters: List[int] = field(default_factory=lambda: [1, 16, 32, 64, 128])
    # kernel_size affects spatial dimensions
    kernel_size: List[Union[int, tuple]] = field(
        default_factory=lambda: [1, 2, 3, 5, (3, 3, 3)]
    )
    # strides affects spatial dimensions
    strides: List[Union[int, tuple]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2, 1), (2, 2, 2)]
    )
    # padding affects spatial dimensions
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    # data_format affects dimension ordering
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )
    # output_padding affects spatial dimensions when strides > 1
    output_padding: List[Optional[Union[int, tuple]]] = field(
        default_factory=lambda: [None, 0, 1, 2, (1, 0, 1)]
    )
    # dilation_rate affects spatial dimensions
    dilation_rate: List[Union[int, tuple]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2, 1), (2, 2, 2)]
    )