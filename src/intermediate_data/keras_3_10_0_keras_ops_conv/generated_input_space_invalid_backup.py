import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.randn(2, 5, 5, 3).astype(np.float32),
    "kernel": np.random.randn(3, 3, 3, 4).astype(np.float32),
    "strides": 1,
    "padding": "same",
    "data_format": "channels_last",
    "dilation_rate": 1
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Class containing all parameters that affect the shape of the output tensor
    """
    kernel: List[np.ndarray] = field(default_factory=lambda: [
        np.random.randn(1, 3, 4).astype(np.float32),      # 1D kernel
        np.random.randn(3, 3, 3, 4).astype(np.float32),   # 2D 3x3 kernel
        np.random.randn(5, 5, 3, 4).astype(np.float32),   # 2D 5x5 kernel
        np.random.randn(3, 3, 3, 8).astype(np.float32),   # Different output channels
        np.random.randn(1, 1, 3, 4).astype(np.float32)    # 2D 1x1 kernel
    ])
    
    strides: List[Union[int, Tuple[int, ...]]] = field(default_factory=lambda: [
        1,
        2,
        (1, 2),
        (2, 1),
        (2, 2)
    ])
    
    padding: List[str] = field(default_factory=lambda: [
        "valid",
        "same"
    ])
    
    data_format: List[str] = field(default_factory=lambda: [
        "channels_last",
        "channels_first"
    ])
    
    dilation_rate: List[Union[int, Tuple[int, ...]]] = field(default_factory=lambda: [
        1,
        2,
        (1, 2),
        (2, 1),
        (2, 2)
    ])