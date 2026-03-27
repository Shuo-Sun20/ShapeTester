import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional, Union

# 1. Define a valid test case
valid_test_case = {
    "inputs": tf.random.normal(shape=[4, 32, 32, 3]),
    "window_shape": [2, 2],
    "pooling_type": "AVG",
    "strides": [2, 2],
    "padding": "VALID",
    "data_format": "NHWC",
    "dilations": [1, 1],
    "name": None
}

# 2 & 3. Parameters affecting output shape (except inputs) and their value spaces
@dataclass
class InputSpace:
    # window_shape: List of ints (spatial dimensions)
    window_shape: List[List[int]] = field(
        default_factory=lambda: [
            [1, 1],  # Minimum window
            [2, 2],  # Small window
            [4, 4],  # Medium window
            [6, 6],  # Larger window
            [8, 8]   # Maximum typical window
        ]
    )
    
    # strides: Sequence of ints (optional)
    strides: List[Optional[List[int]]] = field(
        default_factory=lambda: [
            None,        # Default ([1, 1])
            [1, 1],      # Unit stride
            [2, 2],      # Common stride
            [3, 3],      # Larger stride
            [1, 2]       # Non-square stride
        ]
    )
    
    # padding: String with two possible values
    padding: List[str] = field(
        default_factory=lambda: ["SAME", "VALID"]
    )
    
    # data_format: String with possible channel positions
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [
            None,     # Default (NHWC)
            "NHWC",   # Channels last
            "NCHW"    # Channels first
        ]
    )
    
    # dilations: Sequence of ints (optional)
    dilations: List[Optional[List[int]]] = field(
        default_factory=lambda: [
            None,        # Default ([1, 1])
            [1, 1],      # No dilation
            [2, 2],      # Dilation 2
            [3, 3],      # Dilation 3
            [1, 2]       # Non-square dilation
        ]
    )