import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case variable
batch_size = 2
height, width = 5, 5
num_channels = 3
depth_multiplier = 1
num_output_channels = 4

np.random.seed(42)
inputs = np.random.randn(batch_size, height, width, num_channels).astype(np.float32)
depthwise_kernel = np.random.randn(3, 3, num_channels, depth_multiplier).astype(np.float32)
pointwise_kernel = np.random.randn(1, 1, num_channels * depth_multiplier, num_output_channels).astype(np.float32)

valid_test_case = {
    "inputs": inputs,
    "depthwise_kernel": depthwise_kernel,
    "pointwise_kernel": pointwise_kernel,
    "strides": 1,
    "padding": "valid",
    "data_format": "channels_last",
    "dilation_rate": 1
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    strides: List[Union[int, Tuple[int, ...]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2), (2, 3), (3, 3), (2, 1)]
    )
    padding: List[str] = field(
        default_factory=lambda: ["valid", "same"]
    )
    data_format: List[Union[str, None]] = field(
        default_factory=lambda: ["channels_last", "channels_first", None]
    )
    dilation_rate: List[Union[int, Tuple[int, ...]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 2), (2, 1), (2, 2), (3, 3)]
    )