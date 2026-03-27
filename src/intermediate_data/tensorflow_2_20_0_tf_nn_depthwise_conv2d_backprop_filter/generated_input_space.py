import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case based on the example
valid_test_case = {
    'inputs': [tf.convert_to_tensor(np.random.randn(2, 8, 8, 3).astype(np.float32)),
               tf.convert_to_tensor(np.random.randn(2, 4, 4, 6).astype(np.float32))],
    'filter_sizes': tf.constant([3, 3, 3, 2], dtype=tf.int32),
    'strides': [1, 2, 2, 1],
    'padding': "SAME",
    'data_format': "NHWC",
    'dilations': [1, 1, 1, 1],
    'name': None
}

# 2. & 3. Parameters affecting output shape (except inputs): filter_sizes, strides, padding, data_format, dilations
# Note: The output shape is determined by filter_sizes parameter directly

@dataclass
class InputSpace:
    # filter_sizes: 4-element list [filter_height, filter_width, in_channels, depthwise_multiplier]
    # Each dimension is discrete. We choose 5 typical combinations
    filter_sizes: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],        # Minimal size
        [3, 3, 3, 1],        # Typical square filter
        [3, 3, 3, 2],        # With depth multiplier
        [5, 5, 16, 1],       # Larger filter, more channels
        [7, 7, 32, 2]        # Large filter with multiplier
    ])
    
    # strides: 4-element list [batch, height, width, depth]
    # Typically only height/width stride > 1
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],        # No stride
        [1, 2, 2, 1],        # Common stride of 2
        [1, 2, 1, 1],        # Different horizontal/vertical stride
        [1, 1, 2, 1],        # Different horizontal/vertical stride
        [1, 3, 3, 1]         # Larger stride
    ])
    
    # padding: string or explicit padding list
    # For simplicity, using string padding
    padding: List[Union[str, List[List[int]]]] = field(default_factory=lambda: [
        "VALID",
        "SAME",
        [[0, 0], [1, 1], [1, 1], [0, 0]],  # Explicit padding for NHWC
        [[0, 0], [2, 2], [2, 2], [0, 0]],  # Larger explicit padding
        [[0, 0], [0, 0], [1, 2], [0, 0]]   # Asymmetric explicit padding
    ])
    
    # data_format: string options
    data_format: List[str] = field(default_factory=lambda: [
        "NHWC",
        "NCHW"
    ])
    
    # dilations: 4-element list [batch, height, width, depth]
    dilations: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],        # No dilation
        [1, 2, 2, 1],        # Dilated convolution
        [1, 1, 2, 1],        # Different horizontal/vertical dilation
        [1, 2, 1, 1],        # Different horizontal/vertical dilation
        [1, 3, 3, 1]         # Larger dilation
    ])