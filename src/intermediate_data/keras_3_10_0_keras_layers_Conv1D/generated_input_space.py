import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Optional

# 1. Define valid_test_case
valid_test_case = {
    "filters": 32,
    "kernel_size": 3,
    "strides": 1,
    "padding": "valid",
    "data_format": None,
    "dilation_rate": 1,
    "groups": 1,
    "activation": 'relu',
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "inputs": np.random.rand(4, 10, 128).astype(np.float32)
}

# 2. Parameters affecting output shape: filters, kernel_size, strides, padding, data_format, dilation_rate
# 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # filters: int (discrete, typical range for 1D conv)
    filters: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64, 128, 256])
    
    # kernel_size: int or tuple of 1 integer (discrete, typical values)
    kernel_size: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 3, 5, 7, 11, 21])
    
    # strides: int or tuple of 1 integer (discrete, boundary and typical values)
    strides: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5, 10])
    
    # padding: string (discrete, all possible values)
    padding: List[str] = field(default_factory=lambda: ["valid", "same", "causal"])
    
    # data_format: string or None (discrete, all possible values)
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    
    # dilation_rate: int or tuple of 1 integer (discrete, boundary and typical values)
    dilation_rate: List[Union[int, List[int]]] = field(default_factory=lambda: [1, 2, 3, 4, 5, 10])