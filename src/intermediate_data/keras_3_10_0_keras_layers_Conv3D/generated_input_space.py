import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    'inputs': np.random.rand(4, 10, 10, 10, 128),
    'filters': 32,
    'kernel_size': 3,
    'strides': (1, 1, 1),
    'padding': 'valid',
    'data_format': None,
    'dilation_rate': (1, 1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
    'bias_initializer': 'zeros',
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None
}

# Task 4: Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    # Task 2 identified parameters: filters, kernel_size, strides, padding, data_format, dilation_rate
    
    # filters: positive integer (output channels)
    # Value space: [1, 16, 32, 64, 128, 256]
    filters: List[int] = field(default_factory=lambda: [1, 16, 32, 64, 128, 256])
    
    # kernel_size: int or tuple of 3 ints
    # Value space: [1, 3, 5, 7, (3,3,3), (5,5,5)]
    kernel_size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 3, 5, 7, (3, 3, 3), (5, 5, 5)]
    )
    
    # strides: int or tuple of 3 ints
    # Value space: [1, 2, 3, (1,1,1), (2,2,2), (1,2,3)]
    strides: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 1, 1), (2, 2, 2), (1, 2, 3)]
    )
    
    # padding: string, either "valid" or "same"
    # Value space: ["valid", "same"]
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    
    # data_format: string, either None (default), "channels_last", or "channels_first"
    # Value space: [None, "channels_last", "channels_first"]
    data_format: List[Union[None, str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )
    
    # dilation_rate: int or tuple of 3 ints
    # Value space: [1, 2, 3, (1,1,1), (2,2,2), (1,2,3)]
    dilation_rate: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 2, 3, (1, 1, 1), (2, 2, 2), (1, 2, 3)]
    )