import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define valid_test_case
valid_test_case = {
    'inputs': np.random.rand(4, 10, 10, 12).astype(np.float32),
    'kernel_size': 3,
    'strides': (1, 1),
    'padding': 'valid',
    'depth_multiplier': 1,
    'data_format': None,
    'dilation_rate': (1, 1),
    'activation': 'relu',
    'use_bias': True,
    'depthwise_initializer': 'glorot_uniform',
    'bias_initializer': 'zeros',
    'depthwise_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'depthwise_constraint': None,
    'bias_constraint': None,
    'training': None
}

# 2. Parameters affecting output shape:
# kernel_size, strides, padding, depth_multiplier, data_format, dilation_rate

@dataclass
class InputSpace:
    # kernel_size: discrete values (int or tuple)
    kernel_size: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,
        3,
        5,
        (1, 1),
        (3, 3)
    ])
    
    # strides: discrete values (int or tuple)
    strides: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,
        (1, 1),
        2,
        (2, 2),
        (1, 2)
    ])
    
    # padding: discrete values (two possible strings)
    padding: List[str] = field(default_factory=lambda: [
        "valid",
        "same"
    ])
    
    # depth_multiplier: discrete integer values
    depth_multiplier: List[int] = field(default_factory=lambda: [
        1,
        2,
        3,
        4,
        5
    ])
    
    # data_format: discrete values (string or None)
    data_format: List[Union[str, None]] = field(default_factory=lambda: [
        None,
        "channels_last",
        "channels_first"
    ])
    
    # dilation_rate: discrete values (int or tuple)
    dilation_rate: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,
        (1, 1),
        2,
        (2, 2),
        (1, 2)
    ])