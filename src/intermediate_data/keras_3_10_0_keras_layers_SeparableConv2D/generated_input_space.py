import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

valid_test_case = {
    'inputs': np.random.rand(2, 32, 32, 16).astype(np.float32),
    'filters': 32,
    'kernel_size': (3, 3),
    'strides': (2, 2),
    'padding': "same",
    'data_format': None,
    'dilation_rate': (1, 1),
    'depth_multiplier': 1,
    'activation': "relu",
    'use_bias': True,
    'depthwise_initializer': "glorot_uniform",
    'pointwise_initializer': "glorot_uniform",
    'bias_initializer': "zeros",
    'depthwise_regularizer': None,
    'pointwise_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'depthwise_constraint': None,
    'pointwise_constraint': None,
    'bias_constraint': None
}

@dataclass
class InputSpace:
    filters: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64, 128])
    kernel_size: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1, 2, 3, 5, 7,
        (1, 1), (2, 2), (3, 3), (5, 5), (7, 7),
        (1, 3), (3, 1), (2, 5), (5, 2)
    ])
    strides: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1, 2, 3, 4, 5,
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)
    ])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    dilation_rate: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1, 2, 3, 4, 5,
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)
    ])