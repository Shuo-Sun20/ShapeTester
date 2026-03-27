import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional

# 1. Define valid_test_case with all parameters
x = np.random.rand(4, 10, 12).astype(np.float32)
valid_test_case = {
    "filters": 3,
    "kernel_size": 4,
    "inputs": x,
    "strides": 3,
    "padding": "same",
    "data_format": None,
    "dilation_rate": 1,
    "depth_multiplier": 1,
    "activation": "relu",
    "use_bias": True,
    "depthwise_initializer": "glorot_uniform",
    "pointwise_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "depthwise_regularizer": None,
    "pointwise_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "depthwise_constraint": None,
    "pointwise_constraint": None,
    "bias_constraint": None
}

# 2. & 3. Parameters affecting output tensor shape (excluding "inputs"):
# - filters: int, positive
# - kernel_size: int or tuple/list of 1 int, positive
# - strides: int or tuple/list of 1 int, positive
# - padding: string, discrete ["valid", "same"]
# - data_format: string or None, discrete [None, "channels_last", "channels_first"]
# - dilation_rate: int or tuple/list of 1 int, positive

@dataclass
class InputSpace:
    filters: List[int] = field(default_factory=lambda: [1, 2, 3, 8, 16, 32, 64])
    kernel_size: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7, 11])
    strides: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    dilation_rate: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 7])