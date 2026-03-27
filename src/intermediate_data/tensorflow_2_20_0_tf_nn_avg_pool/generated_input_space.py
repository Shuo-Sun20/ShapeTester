import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# Task 1: Define valid_test_case
np.random.seed(42)
batch_size, height, width, channels = 2, 4, 4, 3
input_shape = [batch_size, height, width, channels]
random_tensor = np.random.rand(*input_shape).astype(np.float32)

valid_test_case = {
    "inputs": [random_tensor],
    "ksize": [1, 2, 2, 1],
    "strides": [1, 1, 1, 1],
    "padding": "SAME",
    "data_format": "NHWC",
    "name": None
}

# Task 4: Define InputSpace class
@dataclass
class InputSpace:
    ksize: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],        # Min pooling size
        [1, 2, 2, 1],        # Example value
        [1, 3, 3, 1],        # Intermediate value
        [1, 4, 4, 1],        # Max for 4x4 input
        [1, 2, 3, 1]         # Asymmetric case
    ])
    
    strides: List[List[int]] = field(default_factory=lambda: [
        [1, 1, 1, 1],        # No striding
        [1, 2, 2, 1],        # 2x stride
        [1, 3, 3, 1],        # 3x stride
        [1, 1, 2, 1],        # Horizontal stride only
        [1, 2, 1, 1]         # Vertical stride only
    ])
    
    padding: List[str] = field(default_factory=lambda: [
        "VALID",
        "SAME"
    ])
    
    data_format: List[str] = field(default_factory=lambda: [
        "NHWC",
        "NCHW"
    ])